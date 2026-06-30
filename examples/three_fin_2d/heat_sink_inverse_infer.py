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
import sys
import warnings

import paddle
import numpy as np
from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node


@modulus.sym.main(config_path="conf_inverse", config_name="config")
def run(cfg: ModulusConfig) -> None:
    nu, D = Symbol("nu"), Symbol("D")

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=1.0, dim=2, time=False)
    ade = AdvectionDiffusion(T="c", rho=1.0, D=D, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    heat_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    invert_net_nu = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("nu")],
        cfg=cfg.arch.fully_connected,
    )
    invert_net_D = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("D")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ns.make_nodes(
            detach_names=[
                "u",
                "u__x",
                "u__x__x",
                "u__y",
                "u__y__y",
                "v",
                "v__x",
                "v__x__x",
                "v__y",
                "v__y__y",
                "p",
                "p__x",
                "p__y",
            ]
        )
        + ade.make_nodes(
            detach_names=["u", "v", "c", "c__x", "c__y", "c__x__x", "c__y__y"]
        )
        + [flow_net.make_node(name="flow_network")]
        + [heat_net.make_node(name="heat_network")]
        + [invert_net_nu.make_node(name="invert_nu_network")]
        + [invert_net_D.make_node(name="invert_D_network")]
    )

    base_temp = 293.498

    # OpenFOAM data
    file_path = "openfoam/heat_sink_Pr5_clipped2.csv"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )
        sys.exit()

    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "U:0": "u",
        "U:1": "v",
        "p": "p",
        "T": "c",
    }
    openfoam_var = csv_to_dict(
        to_absolute_path("openfoam/heat_sink_Pr5_clipped2.csv"), mapping
    )
    openfoam_var["c"] = openfoam_var["c"] / base_temp - 1.0
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "p", "c"]
    }
    openfoam_outvar_numpy["continuity"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["momentum_x"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["momentum_y"] = np.zeros_like(openfoam_outvar_numpy["u"])
    openfoam_outvar_numpy["advection_diffusion_c"] = np.zeros_like(
        openfoam_outvar_numpy["u"]
    )

    # make domain
    domain = Domain()

    # interior
    data = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        outvar=openfoam_outvar_numpy,
        batch_size=cfg.batch_size.data,
        num_workers=0,
    )
    domain.add_constraint(data, "interior_data")

    # add monitors
    monitor = PointwiseMonitor(
        openfoam_invar_numpy,
        output_names=["nu"],
        metrics={"mean_nu": lambda var: paddle.mean(var["nu"])},
        nodes=nodes,
    )
    domain.add_monitor(monitor)

    monitor = PointwiseMonitor(
        openfoam_invar_numpy,
        output_names=["D"],
        metrics={"mean_D": lambda var: paddle.mean(var["D"])},
        nodes=nodes,
    )
    domain.add_monitor(monitor)

    # create solver and initialize models
    slv = Solver(cfg, domain)
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
