"""Taylor Green inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, sin, cos
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.moving_time_window import MovingTimeWindowArch
from modulus.sym.domain.constraint import (
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}

    ns = NavierStokes(nu=0.002, rho=1.0, dim=3, time=True)

    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    channel_length = (0.0, 2 * np.pi)
    channel_width = (0.0, 2 * np.pi)
    channel_height = (0.0, 2 * np.pi)
    box_bounds = {x: channel_length, y: channel_width, z: channel_height}

    rec = Box(
        (channel_length[0], channel_width[0], channel_height[0]),
        (channel_length[1], channel_width[1], channel_height[1]),
    )

    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        periodicity={"x": channel_length, "y": channel_width, "z": channel_height},
        layer_size=256,
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)
    nodes = ns.make_nodes() + [time_window_net.make_node(name="time_window_network")]

    # Use ic_domain setup for inference benchmark
    domain = Domain()
    ic = PointwiseInteriorConstraint(
        nodes=nodes, geometry=rec,
        outvar={
            "u": sin(x) * cos(y) * cos(z),
            "v": -cos(x) * sin(y) * cos(z),
            "w": 0,
            "p": 1.0 / 16 * (cos(2 * x) + cos(2 * y)) * (cos(2 * z) + 2),
        },
        batch_size=cfg.batch_size.initial_condition,
        bounds=box_bounds,
        lambda_weighting={"u": 100, "v": 100, "w": 100, "p": 100},
        parameterization={t_symbol: 0},
        loss=modulus.sym.loss.PointwiseLossNorm(name="ic1"),
    )
    domain.add_constraint(ic, name="ic")
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        bounds=box_bounds,
        batch_size=4094,
        parameterization=time_range,
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior"),
    )
    domain.add_constraint(interior, name="interior")

    slv = Solver(cfg, domain)
    slv.saveable_models = slv.get_saveable_models()
    slv.global_optimizer_model = slv.create_global_optimizer_model()
    from modulus.sym.trainer import Trainer
    Trainer._load_model(
        slv.initialization_network_dir, slv.network_dir,
        slv.saveable_models, 0, slv.log, slv.place,
    )

    warmup_iters = int(os.getenv("INFER_WARMUP", "10"))
    bench_iters = int(os.getenv("INFER_STEPS", "100"))
    for model in slv.saveable_models:
        model.eval()
    slv.load_data()
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
            vals = [l[key] for l in all_losses]
            avg_val = np.mean(vals)
            total_loss += avg_val
            print(f"  loss: {key}: {avg_val:.10f}")
        print(f"  loss: total: {total_loss:.10f}")


if __name__ == "__main__":
    run()
