"""Bracket inference benchmark script (self-contained)."""
import os
import time
import warnings

import numpy as np
from sympy import Symbol, Eq, And
import paddle

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box, Cylinder
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.linear_elasticity import LinearElasticity


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    nu = 0.3
    E = 100e9
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    mu_c = 0.01 * mu
    lambda_ = lambda_ / mu_c
    mu = mu / mu_c
    characteristic_length = 1.0
    characteristic_displacement = 1e-4
    sigma_normalization = characteristic_length / (characteristic_displacement * mu_c)
    T = -4e4 * sigma_normalization

    le = LinearElasticity(lambda_=lambda_, mu=mu, dim=3)
    disp_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w")],
        cfg=cfg.arch.fully_connected,
    )
    stress_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[
            Key("sigma_xx"), Key("sigma_yy"), Key("sigma_zz"),
            Key("sigma_xy"), Key("sigma_xz"), Key("sigma_yz"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        le.make_nodes()
        + [disp_net.make_node(name="displacement_network")]
        + [stress_net.make_node(name="stress_network")]
    )

    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    support_origin = (-1, -1, -1)
    support_dim = (0.25, 2, 2)
    bracket_origin = (-0.75, -1, -0.1)
    bracket_dim = (1.75, 2, 0.2)
    cylinder_radius = 0.1
    cylinder_height = 2.0
    aux_lower_origin = (-0.75, -1, -0.1 - cylinder_radius)
    aux_lower_dim = (cylinder_radius, 2, cylinder_radius)
    aux_upper_origin = (-0.75, -1, 0.1)
    aux_upper_dim = (cylinder_radius, 2, cylinder_radius)
    cylinder_lower_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_upper_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_hole_radius = 0.7
    cylinder_hole_height = 0.5
    cylinder_hole_center = (0.125, 0, 0)

    support = Box(support_origin, (support_origin[0] + support_dim[0], support_origin[1] + support_dim[1], support_origin[2] + support_dim[2]))
    bracket = Box(bracket_origin, (bracket_origin[0] + bracket_dim[0], bracket_origin[1] + bracket_dim[1], bracket_origin[2] + bracket_dim[2]))
    aux_lower = Box(aux_lower_origin, (aux_lower_origin[0] + aux_lower_dim[0], aux_lower_origin[1] + aux_lower_dim[1], aux_lower_origin[2] + aux_lower_dim[2]))
    aux_upper = Box(aux_upper_origin, (aux_upper_origin[0] + aux_upper_dim[0], aux_upper_origin[1] + aux_upper_dim[1], aux_upper_origin[2] + aux_upper_dim[2]))
    cylinder_lower = Cylinder(cylinder_lower_center, cylinder_radius, cylinder_height)
    cylinder_upper = Cylinder(cylinder_upper_center, cylinder_radius, cylinder_height)
    cylinder_hole = Cylinder(cylinder_hole_center, cylinder_hole_radius, cylinder_hole_height)
    cylinder_lower = cylinder_lower.rotate(np.pi / 2, "x")
    cylinder_upper = cylinder_upper.rotate(np.pi / 2, "x")
    cylinder_lower = cylinder_lower.translate([0, 0, -0.1 - cylinder_radius])
    cylinder_upper = cylinder_upper.translate([0, 0, 0.1 + cylinder_radius])
    curve_lower = aux_lower - cylinder_lower
    curve_upper = aux_upper - cylinder_upper
    geo = support + bracket + curve_lower + curve_upper - cylinder_hole

    bounds_support_x = (-1, -0.65)
    bounds_support_y = (-1, 1)
    bounds_support_z = (-1, 1)
    bounds_bracket_x = (-0.65, 1)
    bounds_bracket_y = (-1, 1)
    bounds_bracket_z = (-0.1, 0.1)

    domain = Domain()
    backBC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.backBC,
        lambda_weighting={"u": 10, "v": 10, "w": 10},
        criteria=Eq(x, support_origin[0]),
        loss=modulus.sym.loss.PointwiseLossNorm(name="backBC"),
    )
    domain.add_constraint(backBC, "backBC")
    frontBC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"traction_x": 0, "traction_y": 0, "traction_z": T},
        batch_size=cfg.batch_size.frontBC,
        criteria=Eq(x, bracket_origin[0] + bracket_dim[0]),
        loss=modulus.sym.loss.PointwiseLossNorm(name="frontBC"),
    )
    domain.add_constraint(frontBC, "frontBC")
    surfaceBC = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo,
        outvar={"traction_x": 0, "traction_y": 0, "traction_z": 0},
        batch_size=cfg.batch_size.surfaceBC,
        criteria=And((x > support_origin[0]), (x < bracket_origin[0] + bracket_dim[0])),
        loss=modulus.sym.loss.PointwiseLossNorm(name="surfaceBC"),
    )
    domain.add_constraint(surfaceBC, "surfaceBC")

    interior_outvar = {
        "equilibrium_x": 0.0, "equilibrium_y": 0.0, "equilibrium_z": 0.0,
        "stress_disp_xx": 0.0, "stress_disp_yy": 0.0, "stress_disp_zz": 0.0,
        "stress_disp_xy": 0.0, "stress_disp_xz": 0.0, "stress_disp_yz": 0.0,
    }
    interior_lambda = {
        "equilibrium_x": Symbol("sdf"), "equilibrium_y": Symbol("sdf"), "equilibrium_z": Symbol("sdf"),
        "stress_disp_xx": Symbol("sdf"), "stress_disp_yy": Symbol("sdf"), "stress_disp_zz": Symbol("sdf"),
        "stress_disp_xy": Symbol("sdf"), "stress_disp_xz": Symbol("sdf"), "stress_disp_yz": Symbol("sdf"),
    }
    interior_support = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo, outvar=interior_outvar,
        batch_size=cfg.batch_size.interior_support,
        bounds={x: bounds_support_x, y: bounds_support_y, z: bounds_support_z},
        lambda_weighting=interior_lambda,
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior_support"),
    )
    domain.add_constraint(interior_support, "interior_support")
    interior_bracket = PointwiseInteriorConstraint(
        nodes=nodes, geometry=geo, outvar=interior_outvar,
        batch_size=cfg.batch_size.interior_bracket,
        bounds={x: bounds_bracket_x, y: bounds_bracket_y, z: bounds_bracket_z},
        lambda_weighting=interior_lambda,
        loss=modulus.sym.loss.PointwiseLossNorm(name="interior_bracket"),
    )
    domain.add_constraint(interior_bracket, "interior_bracket")

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
