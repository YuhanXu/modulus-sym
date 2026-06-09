"""Wave inverse inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, Function, Number, sin
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseConstraint
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE


class WaveEquation1D(PDE):
    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        x = Symbol("x")
        t = Symbol("t")
        input_variables = {"x": x, "t": t}
        u = Function("u")(*input_variables)
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
        self.equations = {}
        self.equations["wave_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)


@modulus.sym.main(config_path="conf", config_name="config_inverse")
def run(cfg: ModulusConfig) -> None:
    we = WaveEquation1D(c="c")
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    invert_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        we.make_nodes(detach_names=["u__x", "u__x__x", "u__t__t"])
        + [wave_net.make_node(name="wave_network")]
        + [invert_net.make_node(name="invert_network")]
    )

    L = float(np.pi)
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))
    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": u, "wave_equation": np.zeros_like(u)}

    domain = Domain()
    data = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar_numpy,
        outvar=outvar_numpy,
        batch_size=cfg.batch_size.data,
        num_workers=0,
    )
    domain.add_constraint(data, "interior_data")

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
