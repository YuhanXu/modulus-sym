"""Spring mass ODE inference benchmark script (self-contained)."""
import os
import sys
import time

import numpy as np
from sympy import Symbol, Function, Number
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE


class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self, k=(2, 1, 1, 2), m=(1, 1, 1)):
        k1, k2, k3, k4 = [Number(ki) for ki in k]
        m1, m2, m3 = [Number(mi) for mi in m]
        t = Symbol("t")
        input_variables = {"t": t}
        x1 = Function("x1")(*input_variables)
        x2 = Function("x2")(*input_variables)
        x3 = Function("x3")(*input_variables)
        self.equations = {}
        self.equations["ode_x1"] = m1 * (x1.diff(t)).diff(t) + k1 * x1 - k2 * (x2 - x1)
        self.equations["ode_x2"] = m2 * (x2.diff(t)).diff(t) + k2 * (x2 - x1) - k3 * (x3 - x2)
        self.equations["ode_x3"] = m3 * (x3.diff(t)).diff(t) + k3 * (x3 - x2) + k4 * x3


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    sm = SpringMass(k=(2, 1, 1, 2), m=(1, 1, 1))
    sm_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("x1"), Key("x2"), Key("x3")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sm.make_nodes() + [sm_net.make_node(name="spring_mass_network")]

    geo = Point1D(0)
    t_max = 10.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, t_max)}

    domain = Domain()
    IC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"x1": 1.0, "x2": 0, "x3": 0, "x1__t": 0, "x2__t": 0, "x3__t": 0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x1__t": 1.0, "x2__t": 1.0, "x3__t": 1.0},
        parameterization={t_symbol: 0},
        loss=modulus.sym.loss.PointwiseLossNorm(name="IC"),
    )
    domain.add_constraint(IC, name="IC")
    interior = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"ode_x1": 0.0, "ode_x2": 0.0, "ode_x3": 0.0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior"),
    )
    domain.add_constraint(interior, "interior")

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
