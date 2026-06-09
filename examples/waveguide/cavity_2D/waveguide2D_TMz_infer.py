"""Waveguide cavity 2D TMz inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, pi, sin, Number, Eq
from sympy.logic.boolalg import Or
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pdes.wave_equation import HelmholtzEquation
from modulus.sym.eq.pdes.navier_stokes import GradNormal


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    height = 2
    width = 2
    x, y = Symbol("x"), Symbol("y")

    eigenmode = [2]
    wave_number = 32.0
    waveguide_port = Number(0)
    for k in eigenmode:
        waveguide_port += sin(k * pi * y / height)

    rec = Rectangle((0, 0), (width, height))

    hm = HelmholtzEquation(u="u", k=wave_number, dim=2)
    gn = GradNormal(T="u", dim=2, time=False)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(int(wave_number) * 2 + 1)]),
        frequencies_params=("axis,diagonal", [i / 2.0 for i in range(int(wave_number) * 2 + 1)]),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = hm.make_nodes() + gn.make_nodes() + [wave_net.make_node(name="wave_network")]

    domain = Domain()
    PEC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=rec,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"u": 100.0},
        criteria=Or(Eq(y, 0), Eq(y, height)),
        loss=modulus.sym.loss.PointwiseLossNorm(name="PEC"),
    )
    domain.add_constraint(PEC, "PEC")
    Waveguide_port = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=rec,
        outvar={"u": waveguide_port},
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"u": 100.0},
        criteria=Eq(x, 0),
        loss=modulus.sym.loss.PointwiseLossNorm(name="Waveguide_port"),
    )
    domain.add_constraint(Waveguide_port, "Waveguide_port")
    ABC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=rec,
        outvar={"normal_gradient_u": 0.0},
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={"normal_gradient_u": 10.0},
        criteria=Eq(x, width),
        loss=modulus.sym.loss.PointwiseLossNorm(name="ABC"),
    )
    domain.add_constraint(ABC, "ABC")
    Interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=rec,
        outvar={"helmholtz": 0.0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: (0, width), y: (0, height)},
        lambda_weighting={"helmholtz": 1.0 / wave_number**2},
        loss=modulus.sym.loss.PointwiseLossNorm(name="Interior"),
    )
    domain.add_constraint(Interior, "Interior")

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
