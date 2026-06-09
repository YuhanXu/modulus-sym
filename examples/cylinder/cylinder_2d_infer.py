"""Cylinder 2D inference benchmark script (self-contained)."""
import os
import time
import warnings

import numpy as np
from sympy import Symbol, Eq
import paddle

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Bounds
from modulus.sym.geometry.primitives_2d import Line, Circle, Channel2D
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    nu = quantity(0.02, "kg/(m*s)")
    rho = quantity(1.0, "kg/m^3")
    inlet_u = quantity(1.0, "m/s")
    inlet_v = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(20, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale**3),
    )

    channel_length = (quantity(-10, "m"), quantity(30, "m"))
    channel_width = (quantity(-10, "m"), quantity(10, "m"))
    cylinder_center = (quantity(0, "m"), quantity(0, "m"))
    cylinder_radius = quantity(0.5, "m")
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    cylinder_center_nd = tuple(map(lambda x: nd.ndim(x), cylinder_center))
    cylinder_radius_nd = nd.ndim(cylinder_radius)

    channel = Channel2D(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
    )
    inlet = Line(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[0], channel_width_nd[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    cylinder = Circle(cylinder_center_nd, cylinder_radius_nd)
    volume_geo = channel - cylinder

    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=2, time=False)
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
        + Scaler(
            ["u", "v", "p"],
            ["u_scaled", "v_scaled", "p_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
    )

    domain = Domain()
    x, y = Symbol("x"), Symbol("y")

    inlet_c = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=inlet,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.inlet,
        loss=modulus.sym.loss.PointwiseLossNorm(name="inlet"),
    )
    domain.add_constraint(inlet_c, "inlet")
    outlet_c = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet,
        loss=modulus.sym.loss.PointwiseLossNorm(name="outlet"),
    )
    domain.add_constraint(outlet_c, "outlet")
    walls = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=channel,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.walls,
        loss=modulus.sym.loss.PointwiseLossNorm(name="walls"),
    )
    domain.add_constraint(walls, "walls")
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=cylinder,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.no_slip,
        loss=modulus.sym.loss.PointwiseLossNorm(name="no_slip"),
    )
    domain.add_constraint(no_slip, "no_slip")
    interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
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
