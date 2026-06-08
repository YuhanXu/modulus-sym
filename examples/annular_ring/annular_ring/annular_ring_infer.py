"""Annular ring inference benchmark script."""
import os
import time
import warnings

from sympy import Symbol, Eq, And
import paddle

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # build model
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # geometry
    channel_length = (-6.732, 6.732)
    channel_width = (-1.0, 1.0)
    outer_cylinder_radius = 2.0
    inner_cylinder_radius = 1.0
    inlet_vel = 1.5
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    outer_circle = Circle((0.0, 0.0), outer_cylinder_radius)
    inner_circle = Circle((0, 0), inner_cylinder_radius)
    geo = (rec + outer_circle) - inner_circle

    # build domain with constraints (needed to construct the Solver graph)
    domain = Domain()
    inlet_sympy = parabola(
        y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": inlet_sympy, "v": 0},
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(x, channel_length[0]),
    )
    domain.add_constraint(inlet, "inlet")
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
        criteria=And((x > channel_length[0]), (x < channel_length[1])),
    )
    domain.add_constraint(no_slip, "no_slip")
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"normal_dot_vel": 2},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # create solver and initialize models
    slv = Solver(cfg, domain)
    slv.saveable_models = slv.get_saveable_models()
    slv.global_optimizer_model = slv.create_global_optimizer_model()
    # load model weights only (no optimizer needed for inference)
    from modulus.sym.trainer import Trainer
    Trainer._load_model(
        slv.initialization_network_dir,
        slv.network_dir,
        slv.saveable_models,
        0,
        slv.log,
        slv.place,
    )

    # === Inference benchmark ===
    warmup_iters = int(os.getenv("INFER_WARMUP", "10"))
    bench_iters = int(os.getenv("INFER_STEPS", "100"))

    # set all models to eval mode
    for model in slv.saveable_models:
        model.eval()
    slv.load_data()

    print(f"[Inference] warmup_iters={warmup_iters}, bench_iters={bench_iters}")

    # warmup (PINN models need grad for PDE derivatives, so no paddle.no_grad())
    for i in range(warmup_iters):
        slv.load_data()
        _ = slv.compute_losses(i)

    # benchmark
    paddle.device.cuda.synchronize()
    start = time.perf_counter()
    for i in range(bench_iters):
        slv.load_data()
        _ = slv.compute_losses(i)
    paddle.device.cuda.synchronize()
    end = time.perf_counter()

    avg_latency = (end - start) / bench_iters * 1000  # ms
    print(f"time/iteration: {avg_latency:.4f}")
    print(f"[Inference] avg_latency: {avg_latency:.4f} ms")
    print(f"[Inference] throughput: {bench_iters / (end - start):.2f} iterations/s")
    print(f"[Inference] gpu_memory_peak: {paddle.device.cuda.max_memory_allocated() / (1<<20):.1f} MB")


if __name__ == "__main__":
    run()
