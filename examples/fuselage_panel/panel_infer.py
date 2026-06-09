"""Fuselage panel inference benchmark script (self-contained)."""
import os
import time

import numpy as np
from sympy import Symbol, Eq
import paddle

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    E = 73.0 * 10**9
    nu = 0.33
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu_real = E / (2 * (1 + nu))
    lambda_ = lambda_ / mu_real
    mu = 1.0

    le = LinearElasticityPlaneStress(lambda_=lambda_, mu=mu)
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("sigma_hoop")],
        output_keys=[Key("u"), Key("v"), Key("sigma_xx"), Key("sigma_yy"), Key("sigma_xy")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = le.make_nodes() + [elasticity_net.make_node(name="elasticity_network")]

    x, y, sigma_hoop = Symbol("x"), Symbol("y"), Symbol("sigma_hoop")
    panel_origin = (-0.5, -0.9)
    panel_dim = (1, 1.8)
    window_origin = (-0.125, -0.2)
    window_dim = (0.25, 0.4)
    panel_aux1_origin = (-0.075, -0.2)
    panel_aux1_dim = (0.15, 0.4)
    panel_aux2_origin = (-0.125, -0.15)
    panel_aux2_dim = (0.25, 0.3)
    hr_zone_origin = (-0.2, -0.4)
    hr_zone_dim = (0.4, 0.8)
    circle_nw_center = (-0.075, 0.15)
    circle_ne_center = (0.075, 0.15)
    circle_se_center = (0.075, -0.15)
    circle_sw_center = (-0.075, -0.15)
    circle_radius = 0.05

    panel = Rectangle(panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1]))
    window = Rectangle(window_origin, (window_origin[0] + window_dim[0], window_origin[1] + window_dim[1]))
    panel_aux1 = Rectangle(panel_aux1_origin, (panel_aux1_origin[0] + panel_aux1_dim[0], panel_aux1_origin[1] + panel_aux1_dim[1]))
    panel_aux2 = Rectangle(panel_aux2_origin, (panel_aux2_origin[0] + panel_aux2_dim[0], panel_aux2_origin[1] + panel_aux2_dim[1]))
    hr_zone = Rectangle(hr_zone_origin, (hr_zone_origin[0] + hr_zone_dim[0], hr_zone_origin[1] + hr_zone_dim[1]))
    circle_nw = Circle(circle_nw_center, circle_radius)
    circle_ne = Circle(circle_ne_center, circle_radius)
    circle_se = Circle(circle_se_center, circle_radius)
    circle_sw = Circle(circle_sw_center, circle_radius)
    corners = window - panel_aux1 - panel_aux2 - circle_nw - circle_ne - circle_se - circle_sw
    window = window - corners
    geo = panel - window
    hr_geo = geo & hr_zone

    characteristic_length = panel_dim[0]
    characteristic_disp = 0.001 * window_dim[0]
    sigma_normalization = characteristic_length / (mu_real * characteristic_disp)
    sigma_hoop_lower = 46 * 10**6 * sigma_normalization
    sigma_hoop_upper = 56.5 * 10**6 * sigma_normalization
    sigma_hoop_range = (sigma_hoop_lower, sigma_hoop_upper)
    param_ranges = {sigma_hoop: sigma_hoop_range}

    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])
    hr_bounds_x = (hr_zone_origin[0], hr_zone_origin[0] + hr_zone_dim[0])
    hr_bounds_y = (hr_zone_origin[1], hr_zone_origin[1] + hr_zone_dim[1])

    domain = Domain()
    panel_left = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_left,
        criteria=Eq(x, panel_origin[0]),
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_left"),
    )
    domain.add_constraint(panel_left, "panel_left")
    panel_right = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_right,
        criteria=Eq(x, panel_origin[0] + panel_dim[0]),
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_right"),
    )
    domain.add_constraint(panel_right, "panel_right")
    panel_bottom = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"v": 0.0},
        batch_size=cfg.batch_size.panel_bottom,
        criteria=Eq(y, panel_origin[1]),
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_bottom"),
    )
    domain.add_constraint(panel_bottom, "panel_bottom")
    panel_corner = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.panel_corner,
        criteria=Eq(x, panel_origin[0]) & (y > panel_origin[1]) & (y < panel_origin[1] + 1e-3),
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_corner"),
    )
    domain.add_constraint(panel_corner, "panel_corner")
    panel_top = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": sigma_hoop},
        batch_size=cfg.batch_size.panel_top,
        criteria=Eq(y, panel_origin[1] + panel_dim[1]),
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_top"),
    )
    domain.add_constraint(panel_top, "panel_top")
    panel_window = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=window,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_window,
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="panel_window"),
    )
    domain.add_constraint(panel_window, "panel_window")

    interior_outvar = {
        "equilibrium_x": 0.0, "equilibrium_y": 0.0,
        "stress_disp_xx": 0.0, "stress_disp_yy": 0.0, "stress_disp_xy": 0.0,
    }
    interior_lambda = {
        "equilibrium_x": Symbol("sdf"), "equilibrium_y": Symbol("sdf"),
        "stress_disp_xx": Symbol("sdf"), "stress_disp_yy": Symbol("sdf"), "stress_disp_xy": Symbol("sdf"),
    }
    lr_interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo, outvar=interior_outvar,
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y},
        lambda_weighting=interior_lambda,
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="lr_interior"),
    )
    domain.add_constraint(lr_interior, "lr_interior")
    hr_interior = PointwiseInteriorConstraint(
        nodes=nodes, geometry=hr_geo, outvar=interior_outvar,
        batch_size=cfg.batch_size.hr_interior,
        bounds={x: hr_bounds_x, y: hr_bounds_y},
        lambda_weighting=interior_lambda,
        parameterization=param_ranges,
        loss=modulus.sym.loss.PointwiseLossNorm(name="hr_interior"),
    )
    domain.add_constraint(hr_interior, "hr_interior")

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
