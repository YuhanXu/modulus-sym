"""Sphere surface PDE inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, Function
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D
from modulus.sym.geometry.primitives_3d import Sphere
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE


class SurfacePoisson(PDE):
    name = "SurfacePoisson"

    def __init__(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")
        u = Function("u")(x, y, z)
        self.equations = {}
        self.equations["poisson_u"] = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        self.equations["flux_u"] = normal_x * u.diff(x) + normal_y * u.diff(y) + normal_z * u.diff(z)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    sp = SurfacePoisson()
    poisson_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sp.make_nodes() + [poisson_net.make_node(name="poisson_network")]

    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    geo = Sphere((0, 0, 0), 1)
    p = Point1D(1, parameterization=Parameterization({Parameter("y"): 0, Parameter("z"): 0}))

    domain = Domain()
    surface = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"poisson_u": -18.0 * x * y * z, "flux_u": 0},
        batch_size=cfg.batch_size.surface,
        lambda_weighting={"poisson_u": 1.0, "flux_u": 1.0},
        loss=modulus.sym.loss.PointwiseLossNorm(name="surface"),
    )
    domain.add_constraint(surface, "surface")
    point = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=p,
        outvar={"u": 0.0},
        batch_size=2,
        lambda_weighting={"u": 1.0},
        loss=modulus.sym.loss.PointwiseLossNorm(name="point"),
    )
    domain.add_constraint(point, "point")

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
