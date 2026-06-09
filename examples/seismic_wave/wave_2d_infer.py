"""Seismic wave 2D inference benchmark script (self-contained)."""
import os
import sys
import time
import warnings

import numpy as np
from sympy import Symbol, Function, Number
import paddle

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.key import Key
from modulus.sym.eq.pdes.wave_equation import WaveEquation
from modulus.sym.eq.pde import PDE


class OpenBoundary(PDE):
    name = "OpenBoundary"

    def __init__(self, u="u", c="c", dim=2, time=True):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")
        t = Symbol("t")
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif dim == 2:
            input_variables.pop("z")
        if not time:
            input_variables.pop("t")
        u = Function(u)(*input_variables)
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
        self.equations = {}
        self.equations["open_boundary"] = (
            u.diff(t) + normal_x * c * u.diff(x) + normal_y * c * u.diff(y)
        )
        if dim == 3:
            self.equations["open_boundary"] += normal_z * c * u.diff(z)


def read_wf_data(time_ms, dLen):
    file_path = "Training_data"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(f"Directory {file_path} does not exist.")
        sys.exit()
    wf_filename = to_absolute_path(f"Training_data/wf_{int(time_ms):04d}ms.npz")
    wave = np.load(wf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, wave.shape[0]),
        np.linspace(0, dLen, wave.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time_ms * 0.001)
    outvar = {}
    outvar["u"] = np.expand_dims(wave.flatten(), axis=-1)
    return invar, outvar


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    cfg.arch.fully_connected.layer_size = 128

    we = WaveEquation(u="u", c="c", dim=2, time=True)
    ob = OpenBoundary(u="u", c="c", dim=2, time=True)

    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    speed_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        we.make_nodes(detach_names=["c"])
        + ob.make_nodes(detach_names=["c"])
        + [wave_net.make_node(name="wave_network"), speed_net.make_node(name="speed_network")]
    )

    dLen = 2
    rec = Rectangle((0, 0), (dLen, dLen))
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    time_length = 1
    time_range = {t: (0.15, time_length)}

    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, 2, 512), np.linspace(0, 2, 512), indexing="ij"
    )
    wave_speed_invar = {
        "x": np.expand_dims(mesh_x.flatten(), axis=-1),
        "y": np.expand_dims(mesh_y.flatten(), axis=-1),
    }
    wave_speed_outvar = {"c": np.tanh(80 * (wave_speed_invar["y"] - 1.0)) / 2 + 1.5}

    domain = Domain()
    velocity = PointwiseConstraint.from_numpy(
        nodes=nodes, invar=wave_speed_invar, outvar=wave_speed_outvar, batch_size=1024,
        num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Velocity"),
    )
    domain.add_constraint(velocity, "Velocity")

    batch_size = 1024
    for i, ms in enumerate(np.linspace(150, 300, 4)):
        timestep_invar, timestep_outvar = read_wf_data(ms, dLen)
        lambda_weighting = {"u": np.full_like(timestep_invar["x"], 10.0 / batch_size)}
        timestep = PointwiseConstraint.from_numpy(
            nodes, timestep_invar, timestep_outvar, batch_size,
            lambda_weighting=lambda_weighting, num_workers=0,
            loss=modulus.sym.loss.PointwiseLossNorm(name=f"BC{i:04d}"),
        )
        domain.add_constraint(timestep, f"BC{i:04d}")

    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=rec,
        outvar={"wave_equation": 0},
        batch_size=4096,
        bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"wave_equation": 0.0001},
        parameterization=time_range, num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Interior"),
    )
    domain.add_constraint(interior, "Interior")
    edges = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=rec,
        outvar={"open_boundary": 0},
        batch_size=1024,
        lambda_weighting={"open_boundary": 0.01 * time_length},
        parameterization=time_range, num_workers=0,
        loss=modulus.sym.loss.PointwiseLossNorm(name="Edges"),
    )
    domain.add_constraint(edges, "Edges")

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
