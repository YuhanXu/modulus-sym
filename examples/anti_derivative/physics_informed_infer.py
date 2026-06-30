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

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.deeponet import DeepONetArch
from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.key import Key


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # [init-model]
    # make list of nodes to unroll graph on
    trunk_net = FourierNetArch(
        input_keys=[Key("x")],
        output_keys=[Key("trunk", 128)],
    )
    branch_net = FullyConnectedArch(
        input_keys=[Key("a", 100)],
        output_keys=[Key("branch", 128)],
    )
    deeponet = DeepONetArch(
        output_keys=[Key("u")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )

    nodes = [deeponet.make_node("deepo")]
    # [init-model]

    # [datasets]
    # load training data
    file_path = "data/anti_derivative.npy"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )
        sys.exit()

    data = np.load(to_absolute_path(file_path), allow_pickle=True).item()
    x_train = data["x_train"]
    a_train = data["a_train"]
    u_train = data["u_train"]

    x_r_train = data["x_r_train"]
    a_r_train = data["a_r_train"]
    u_r_train = data["u_r_train"]

    # load test data
    x_test = data["x_test"]
    a_test = data["a_test"]
    u_test = data["u_test"]
    # [datasets]

    # [constraint1]
    # make domain
    domain = Domain()

    # add constraints to domain
    IC = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_train, "x": np.zeros_like(x_train)},
        outvar={"u": np.zeros_like(u_train)},
        batch_size=cfg.batch_size.train,
        loss=modulus.sym.loss.PointwiseLossNorm(name="IC"),
    )
    domain.add_constraint(IC, "IC")
    # [constraint1]

    # [constraint2]
    interior = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_r_train, "x": x_r_train},
        outvar={"u__x": u_r_train},
        batch_size=cfg.batch_size.train,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Residual"),
    )
    domain.add_constraint(interior, "Residual")
    # [constraint2]

    # [validator]
    # add validators
    for k in range(10):
        invar_valid = {
            "a": a_test[k * 100 : (k + 1) * 100],
            "x": x_test[k * 100 : (k + 1) * 100],
        }
        outvar_valid = {"u": u_test[k * 100 : (k + 1) * 100]}
        dataset = DictGridDataset(invar_valid, outvar_valid)

        validator = GridValidator(nodes=nodes, dataset=dataset, plotter=None)
        domain.add_validator(validator, "validator_{}".format(k))
    # [validator]

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
