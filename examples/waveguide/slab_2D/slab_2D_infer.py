"""Waveguide slab 2D inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, Eq, Heaviside, sqrt
from sympy.logic.boolalg import Or
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pdes.wave_equation import HelmholtzEquation
from modulus.sym.eq.pdes.navier_stokes import GradNormal

x, y = Symbol("x"), Symbol("y")


def Laplacian_1D_eig(a, b, N, eps=lambda x: np.ones_like(x), k=3):
    n = N - 2
    h = (b - a) / (N - 1)
    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2
    xv = np.linspace(a, b, num=N)
    M = diags([eps(xv[1:-1])], [0])
    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
    eigvecs /= norm_eigvecs
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), xv.astype(np.float32)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    height = 2
    width = 2
    len_slab = 0.6
    eps0 = 1.0
    eps1 = 2.0
    eps_numpy = lambda yy: np.where(
        np.logical_and(yy > (height - len_slab) / 2, yy < (height + len_slab) / 2),
        eps1, eps0,
    )
    eps_sympy = sqrt(
        eps0 + (Heaviside(y - (height - len_slab) / 2) - Heaviside(y - (height + len_slab) / 2)) * (eps1 - eps0)
    )
    eigvals, eigvecs, yv = Laplacian_1D_eig(0, height, 1000, eps=eps_numpy, k=3)
    yv = yv.reshape((-1, 1))
    wave_number = 16.0
    waveguide_port_invar_numpy = {"x": np.zeros_like(yv), "y": yv}
    waveguide_port_outvar_numpy = {"u": 10 * eigvecs[:, 0:1]}

    rec = Rectangle((0, 0), (width, height))
    hm = HelmholtzEquation(u="u", k=wave_number * eps_sympy, dim=2)
    gn = GradNormal(T="u", dim=2, time=False)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)]),
        frequencies_params=("axis,diagonal", [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)]),
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
    Waveguide_port = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=waveguide_port_invar_numpy,
        outvar=waveguide_port_outvar_numpy,
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"u": np.full_like(yv, 0.5)},
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
