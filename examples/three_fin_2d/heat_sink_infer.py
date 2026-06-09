"""Three fin 2D heat sink inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, Eq
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.key import Key
from modulus.sym.geometry import Parameterization, Parameter


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    heat_sink_origin = (-1, -0.3)
    nr_heat_sink_fins = 3
    gap = 0.15 + 0.1
    heat_sink_length = 1.0
    heat_sink_fin_thickness = 0.1
    inlet_vel = 1.5
    heat_sink_temp = 350
    base_temp = 293.498
    nu = 0.01
    diffusivity = 0.01 / 5

    x, y = Symbol("x"), Symbol("y")

    channel = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    hs_origin = list(heat_sink_origin)
    heat_sink = Rectangle(
        tuple(hs_origin),
        (hs_origin[0] + heat_sink_length, hs_origin[1] + heat_sink_fin_thickness),
    )
    for i in range(1, nr_heat_sink_fins):
        hs_origin[1] = hs_origin[1] + gap
        fin = Rectangle(
            tuple(hs_origin),
            (hs_origin[0] + heat_sink_length, hs_origin[1] + heat_sink_fin_thickness),
        )
        heat_sink = heat_sink + fin
    geo = channel - heat_sink

    inlet_geo = Line(
        (channel_length[0], channel_width[0]), (channel_length[0], channel_width[1]), -1
    )
    outlet_geo = Line(
        (channel_length[1], channel_width[0]), (channel_length[1], channel_width[1]), 1
    )
    x_pos = Parameter("x_pos")
    integral_line = Line(
        (x_pos, channel_width[0]), (x_pos, channel_width[1]), 1,
        parameterization=Parameterization({x_pos: channel_length}),
    )

    ze = ZeroEquation(nu=nu, rho=1.0, dim=2, max_distance=(channel_width[1] - channel_width[0]) / 2)
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=False)
    ade = AdvectionDiffusion(T="c", rho=1.0, D=diffusivity, dim=2, time=False)
    gn_c = GradNormal("c", dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
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
    nodes = (
        ns.make_nodes() + ze.make_nodes()
        + ade.make_nodes(detach_names=["u", "v"])
        + gn_c.make_nodes() + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [heat_net.make_node(name="heat_network")]
    )

    domain = Domain()
    inlet_parabola = parabola(y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel)
    inlet_c = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=inlet_geo,
        outvar={"u": inlet_parabola, "v": 0, "c": 0},
        batch_size=cfg.batch_size.inlet,
        loss=modulus.sym.loss.PointwiseLossNorm(name="inlet"),
    )
    domain.add_constraint(inlet_c, "inlet")
    outlet_c = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=outlet_geo,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        loss=modulus.sym.loss.PointwiseLossNorm(name="outlet"),
    )
    domain.add_constraint(outlet_c, "outlet")
    hs_wall = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=heat_sink,
        outvar={"u": 0, "v": 0, "c": (heat_sink_temp - base_temp) / 273.15},
        batch_size=cfg.batch_size.hs_wall,
        loss=modulus.sym.loss.PointwiseLossNorm(name="heat_sink_wall"),
    )
    domain.add_constraint(hs_wall, "heat_sink_wall")
    channel_wall = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=channel,
        outvar={"u": 0, "v": 0, "normal_gradient_c": 0},
        batch_size=cfg.batch_size.channel_wall,
        loss=modulus.sym.loss.PointwiseLossNorm(name="channel_wall"),
    )
    domain.add_constraint(channel_wall, "channel_wall")
    interior_flow = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_flow,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"), "momentum_x": Symbol("sdf"), "momentum_y": Symbol("sdf"),
        },
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior_flow"),
    )
    domain.add_constraint(interior_flow, "interior_flow")
    interior_heat = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo,
        outvar={"advection_diffusion_c": 0},
        batch_size=cfg.batch_size.interior_heat,
        lambda_weighting={"advection_diffusion_c": 1.0},
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior_heat"),
    )
    domain.add_constraint(interior_heat, "interior_heat")

    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes, geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=integral_criteria,
        loss=modulus.sym.loss.IntegralLossNorm(name="integral_continuity"),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

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
