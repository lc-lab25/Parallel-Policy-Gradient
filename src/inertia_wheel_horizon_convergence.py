#!/usr/bin/env python3
"""Inertia-wheel DEER convergence across horizons T and initialization radii r.

Here r is the initialization radius PER TIME STEP:
    s_k^(0) = r * v_k / ||v_k||_2,  v_k ~ N(0,I_4),  k=1,...,T.
Thus ||s_k^(0)||_2=r, around the equilibrium x*=0. The physical initial
condition x0 remains [pi-0.2, 0, 0, 0]. The initialization uses no reference.
The same directions are shared by all radii and by common horizon prefixes.
The total guess norm is r*sqrt(T); it is NOT a fixed trajectory L2 radius.

The single axis plots the unnormalized trajectory error
    e_i(T,r) = ||s^(i) - x_reference[1:T+1]||_2
against the Newton update i. Each curve is the worst error across the chosen
random seeds. Color identifies T; line style and marker identify r.

Outputs: convergence.png and convergence.pdf (the SAME single-axis plot),
history.csv, summary.csv, and metadata.json in --out. Nonfinite or unfinished
runs are saved and indicated on the plot. The CSV files retain every seed;
no unsuccessful run is discarded. The plot ends a T/r curve when all its
runs finish; already stopped seeds retain their final error in the maximum.

This is empirical evidence for the tested initialization, controller, and
horizons. Increasing the tested grid does not prove a claim for every T.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

import inertia_wheel_projected_deer as model
from deer import deer_alg, get_residual


# =============================================================================
# Experiment configuration
# =============================================================================
HORIZONS = [ 100, 125, 250, 375, 500, 625, 750, 875, 1000]
RADII = [3]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

DEER_MAX_ITERS = 200
DEER_TOL = 1e-8

# State coordinates are [q1, q2, p1, p2]; p1 and p2 are momenta.
X0 = [np.pi - 0.2, 0.0, 0.0, 0.0]
THETA = model.theta_initial

# Anchor the default output directory to the project, regardless of cwd.
RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "inertia_wheel_horizon_convergence_results"
)


def parse_args(argv=None):
    """Use the constants above as defaults, with optional CLI overrides."""
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--horizons", nargs="+", type=int,
                   default=HORIZONS)
    p.add_argument("--radii", "--r-values", nargs="+", type=float,
                   default=RADII)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--max-iters", type=int, default=DEER_MAX_ITERS)
    p.add_argument("--tol", type=float, default=DEER_TOL)
    p.add_argument("--x0", nargs=4, type=float,
                   default=X0, metavar=("Q1", "Q2", "P1", "P2"))
    a = p.parse_args(argv)
    if min(a.horizons) < 2 or min(a.seeds) < 0 or a.max_iters < 1:
        p.error("Require T>=2, seeds>=0, and max-iters>=1.")
    if not np.all(np.isfinite([*a.radii, a.tol, *a.x0])) or min(a.radii) <= 0 or a.tol <= 0:
        p.error("Require positive finite radii and tolerance, and finite x0.")
    a.horizons, a.radii, a.seeds = map(lambda v: sorted(set(v)), (a.horizons, a.radii, a.seeds))
    return a


def stable_norm(values):
    """Euclidean norm with scaling to avoid premature overflow."""
    scale = jnp.max(jnp.abs(values))
    safe = jnp.where(scale > 0, scale, 1.0)
    return scale * jnp.sqrt(jnp.sum((values / safe) ** 2))


@jax.jit
def newton_step(x0, candidate, theta):
    """Perform exactly one full-Jacobian, undamped update with deer_alg."""
    def forward_f(x, _):
        return model.closed_loop_step(x, theta)

    return deer_alg(
        forward_f,
        x0,
        candidate,
        jnp.zeros((len(candidate), 1)),
        num_iters=1,
        full_trace=True,
        quasi=False,
        k=0,
        clip=False,
        reset=False,
        get_lles=False,
    )[1]


@jax.jit
def measure_errors(x0, candidate, reference, theta):
    """Return absolute trajectory-error L2 and dynamics-residual L2 norms."""
    def forward_f(x, _):
        return model.closed_loop_step(x, theta)

    residual = get_residual(
        forward_f, x0, candidate, jnp.zeros((len(candidate), 1))
    )
    return jnp.array([
        stable_norm(candidate - reference),
        stable_norm(residual),
    ])


def simulate_reference(x0, theta, horizon):
    """Call the existing simulation with its horizon setting restored afterward."""
    previous_horizon = model.T_HORIZON
    try:
        model.T_HORIZON = horizon
        # sequential_rollout already uses lax.scan. Calling it directly avoids
        # caching a global T_HORIZON in an additional jit wrapper.
        states = model.sequential_rollout(x0, theta)
        states.block_until_ready()
    finally:
        model.T_HORIZON = previous_horizon

    if states.shape != (horizon + 1, len(x0)):
        raise RuntimeError("The simulation returned an unexpected trajectory shape.")
    if not np.all(np.isfinite(np.asarray(states))):
        raise RuntimeError("The simulation returned a nonfinite reference trajectory.")
    return states[1:]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_convergence(args, history, summary):
    """Exactly one Figure and one Axes; encode T by color and r by style."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    colors = plt.get_cmap("viridis")(np.linspace(0.06, 0.88, len(args.horizons)))
    styles = ["-", "--", "-.", ":", (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    fig, ax = plt.subplots(figsize=(8, 4))
    failed = []
    for ti, T in enumerate(args.horizons):
        for ri, radius in enumerate(args.radii):
            runs = [[row for row in history if (row["T"], row["r"], row["seed"]) == (T, radius, seed)]
                    for seed in args.seeds]
            n = max(len(run) for run in runs)
            # A stopped run's final state is unchanged. Include its measured
            # terminal error while the other seeds continue iterating.
            seed_errors = np.array([[run[min(i, len(run)-1)]["error_l2"]
                                     for i in range(n)] for run in runs])
            worst = np.max(seed_errors, axis=0)  # NaNs propagate: never hide a failed seed.
            values = np.where(np.isfinite(worst) & (worst > 0), worst, np.nan)
            # Explicit logarithmic coordinates avoid tick overflow if a
            # failed experiment produces extremely large finite errors.
            ax.plot(np.arange(n), np.log10(values), color=colors[ti],
                    linestyle=styles[ri % len(styles)], marker=markers[ri % len(markers)],
                    markersize=5, markerfacecolor="none", linewidth=1.5, alpha=0.9)
            group = [row for row in summary if row["T"] == T and row["r"] == radius]
            count = sum(row["status"] != "converged" for row in group)
            if count:
                failed.append(f"T={T:g}, r={radius:g}: {count}/{len(group)}")

    ax.axhline(np.log10(args.tol), color="0.35", ls="--", lw=1.1)
    ax.text(0.025, np.log10(args.tol)+0.13, f"tolerance = {args.tol:g}",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=9, color="0.3")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: rf"$10^{{{y:g}}}$"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Gauss-Newton iteration $i$")
    ax.set_ylabel(r"Trajectory error $\|r^{(i)}_x\|_2$")
    ax.grid(alpha=0.19)
    ax.set_title("Inertia-wheel pendulum: convergence across T\n"
                 f"Maximum error across {len(args.seeds)} seeds; absolute L2 norm", pad=14, fontsize=13)
    # Legends occupy their own figure margin, leaving the data visible.
    horizon_handles = [Line2D([], [], color=colors[i], lw=2, label=f"{T:,}")
                       for i, T in enumerate(args.horizons)]
    radius_handles = [Line2D([], [], color="0.2", lw=1.3,
                            linestyle=styles[i % len(styles)], marker=markers[i % len(markers)],
                            markersize=4, markerfacecolor="none", label=f"{r:g}")
                      for i, r in enumerate(args.radii)]
    fig.legend(handles=horizon_handles, title="Horizon T", loc="upper left",
               bbox_to_anchor=(0.79, 0.85), frameon=False, fontsize=10, title_fontsize=10)
    # fig.legend(handles=radius_handles, title="Radius r (style)", loc="upper left",
    #            bbox_to_anchor=(0.79, 0.43), frameon=False, fontsize=10, title_fontsize=10)
    if failed:
        text = "Unconverged groups: " + "; ".join(failed)
        if len(text) > 160:
            text = f"{len(failed)} T/r groups contain unconverged runs; all are recorded in summary.csv."
        fig.text(0.12, 0.015, text, color="firebrick", fontsize=8, wrap=True)
    fig.subplots_adjust(left=0.115, right=0.775, bottom=0.12, top=0.85)
    fig.savefig(args.out/"convergence.png", dpi=200)
    fig.savefig(args.out/"convergence.pdf")
    plt.close(fig)


def main(argv=None):
    """Run the sweep and save one convergence figure plus its measurements."""
    args = parse_args(argv)
    source = Path(model.__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    theta = jnp.asarray(THETA, dtype=jnp.float64)
    x0 = jnp.asarray(args.x0, dtype=jnp.float64)
    reference_max = simulate_reference(x0, theta, max(args.horizons))

    try:
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = dict(repository="https://github.com/lc-lab25/Parallel-Policy-Gradient",
        commit=commit, source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ["inertia_wheel_projected_deer.py", "deer.py", "lle.py"]},
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        DT=float(model.DT), theta=np.asarray(theta).tolist(),
        jax_version=jax.__version__, numpy_version=np.__version__, devices=[str(d) for d in jax.devices()],
        radius_definition="||s_k^(0)||_2=r for each k>=1, around equilibrium 0; x0 stays fixed",
        simulation="inertia_wheel_projected_deer.sequential_rollout",
        solver="deer.deer_alg, full Jacobian, undamped, one update per call",
        metric="Unnormalized trajectory-error L2 norm, maximum across seeds for each T/r",
        stopping="Both absolute trajectory-error L2 and residual L2 <= tol",
        seed_aggregation="Stopped runs retain their measured final error until the last seed finishes")
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"DT={model.DT:g}, theta={np.asarray(theta)}, x0={np.asarray(x0)}", flush=True)
    print(f"{len(args.horizons)} horizons x {len(args.radii)} radii x {len(args.seeds)} seeds; "
          f"backend={jax.devices()}; float64", flush=True)
    history, summary = [], []
    for T in args.horizons:
        reference = reference_max[:T]
        for seed in args.seeds:
            directions = np.random.default_rng(seed).normal(size=(T, 4))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            directions = jnp.asarray(directions)
            for radius in args.radii:
                candidate = radius*directions
                status = "iteration_limit"
                previous_error = None
                increases = 0
                for i in range(args.max_iters+1):
                    error, residual = map(float, np.asarray(
                        measure_errors(x0, candidate, reference, theta)
                    ))
                    finite = np.isfinite(error) and np.isfinite(residual)
                    if previous_error is not None and finite and previous_error > args.tol:
                        increases += int(error > previous_error*(1+1e-10))
                    history.append(dict(T=T, r=radius, seed=seed, iteration=i,
                                        error_l2=error, residual_l2=residual))
                    if not finite:
                        status = "nonfinite"
                        break
                    if max(error, residual) <= args.tol:
                        status = "converged"
                        break
                    if i < args.max_iters:
                        previous_error = error
                        candidate = newton_step(x0, candidate, theta)
                summary.append(dict(T=T, r=radius, seed=seed, status=status, updates=i,
                    final_error_l2=error, final_residual_l2=residual,
                    error_increases_before_tolerance=increases))
        # Preserve all measurements after each horizon.
        write_csv(args.out/"history.csv", history)
        write_csv(args.out/"summary.csv", summary)
        for radius in args.radii:
            selected = [row for row in summary if row["T"] == T and row["r"] == radius]
            counts = [row["updates"] for row in selected if row["status"] == "converged"]
            spread = f"{min(counts)}-{max(counts)} updates" if counts else "no converged runs"
            print(f"T={T:>7d} r={radius:>5g}: {len(counts)}/{len(selected)} converged; {spread}", flush=True)
    plot_convergence(args, history, summary)
    print(f"Saved one plot (PNG/PDF) and raw measurements to {args.out.resolve()}")


if __name__ == "__main__":
    main()
