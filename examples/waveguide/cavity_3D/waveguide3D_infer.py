import paddle
import time
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from sympy import Symbol, pi, sin, Number, Eq, And

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io.plotter import InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.eq.pdes.electromagnetic import PEC, SommerfeldBC, MaxwellFreqReal
import numpy as np

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # params for domain
    length = 2
    height = 2
    width = 2

    eigenmode = [1]
    wave_number = 16.0  # wave_number = freq/c
    waveguide_port = Number(0)
    for k in eigenmode:
        waveguide_port += sin(k * pi * y / length) * sin(k * pi * z / height)

    # define geometry
    rec = Box((0, 0, 0), (width, length, height))
    # make list of nodes to unroll graph on
    hm = MaxwellFreqReal(k=wave_number)
    pec = PEC()
    pml = SommerfeldBC()
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("ux"), Key("uy"), Key("uz")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(int(wave_number) + 1)]),
        frequencies_params=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number) + 1)],
        ),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = (
        hm.make_nodes()
        + pec.make_nodes()
        + pml.make_nodes()
        + [wave_net.make_node(name="wave_network")]
    )

    waveguide_domain = Domain()

    wall_PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"PEC_x": 0.0, "PEC_y": 0.0, "PEC_z": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"PEC_x": 100.0, "PEC_y": 100.0, "PEC_z": 100.0},
        criteria=And(~Eq(x, 0), ~Eq(x, width)),
        num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="PEC"),
    )

    waveguide_domain.add_constraint(wall_PEC, "PEC")

    Waveguide_port = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"uz": waveguide_port},
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"uz": 100.0},
        criteria=Eq(x, 0),
        num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Waveguide_port"),
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "SommerfeldBC_real_x": 0.0,
            "SommerfeldBC_real_y": 0.0,
            "SommerfeldBC_real_z": 0.0,
        },
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={
            "SommerfeldBC_real_x": 10.0,
            "SommerfeldBC_real_y": 10.0,
            "SommerfeldBC_real_z": 10.0,
        },
        criteria=Eq(x, width),
        num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="ABC"),
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "Maxwell_Freq_real_x": 0,
            "Maxwell_Freq_real_y": 0.0,
            "Maxwell_Freq_real_z": 0.0,
        },
        batch_size=cfg.batch_size.Interior,
        bounds={x: (0, width), y: (0, length), z: (0, height)},
        lambda_weighting={
            "Maxwell_Freq_real_x": 1.0 / wave_number**2,
            "Maxwell_Freq_real_y": 1.0 / wave_number**2,
            "Maxwell_Freq_real_z": 1.0 / wave_number**2,
        },
        fixed_dataset=False,
        num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Interior"),
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add inferencer data
    interior_points = rec.sample_interior(
        10000, bounds={x: (0, width), y: (0, length), z: (0, height)}
    )
    numpy_inference = PointwiseInferencer(
        nodes=nodes,
        invar=interior_points,
        output_names=["ux", "uy", "uz"],
        plotter=InferencerPlotter(),
        batch_size=2048,
    )
    waveguide_domain.add_inferencer(numpy_inference, "Inf" + str(wave_number).zfill(4))

    # create solver and initialize models
    slv = Solver(cfg, waveguide_domain)
    slv.saveable_models = slv.get_saveable_models()
    slv.global_optimizer_model = slv.create_global_optimizer_model()
    from modulus.sym.trainer import Trainer
    Trainer._load_model(
        slv.initialization_network_dir,
        slv.network_dir,
        slv.saveable_models,
        0,
        slv.log,
        slv.place,
    )

    warmup_iters = int(os.getenv("INFER_WARMUP", "10"))
    bench_iters = int(os.getenv("INFER_STEPS", "100"))

    for model in slv.saveable_models:
        model.eval()

    print(f"[Inference] warmup_iters={warmup_iters}, bench_iters={bench_iters}")

    for i in range(warmup_iters):
        slv.load_data()
        _ = slv.compute_losses(i)

    paddle.device.cuda.synchronize()
    start = time.perf_counter()
    for i in range(bench_iters):
        slv.load_data()
        _ = slv.compute_losses(i)
    paddle.device.cuda.synchronize()
    end = time.perf_counter()

    avg_latency = (end - start) / bench_iters * 1000
    print(f"time/iteration: {avg_latency:.4f}")
    print(f"[Inference] avg_latency: {avg_latency:.4f} ms")
    print(f"[Inference] throughput: {bench_iters / (end - start):.2f} iterations/s")
    print(f"[Inference] gpu_memory_peak: {paddle.device.cuda.max_memory_allocated() / (1<<20):.1f} MB")

    if os.getenv("INFER_CHECK_ACCURACY", "0") == "1":
        accuracy_iters = int(os.getenv("INFER_ACCURACY_ITERS", "50"))
        all_losses = []
        for i in range(accuracy_iters):
            slv.load_data()
            losses = slv.compute_losses(i)
            all_losses.append({k: v.item() for k, v in losses.items()})
        loss_keys = all_losses[0].keys()
        print(f"\n[Accuracy] Per-constraint average loss ({accuracy_iters} iters):")
        total_loss = 0.0
        for key in sorted(loss_keys):
            vals = [loss[key] for loss in all_losses]
            avg_val = np.mean(vals)
            total_loss += avg_val
            print(f"  loss: {key}: {avg_val:.10f}")
        print(f"  loss: total: {total_loss:.10f}")


if __name__ == "__main__":
    run()
