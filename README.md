# Parallel Policy-Gradient Methods for Parameter Optimization of Nonlinear Feedback Controllers

This repository contains the code and numerical experiments for **“Parallel Policy-Gradient Methods for Parameter Optimization of Nonlinear Feedback Controllers.”**

We develop a time-parallel framework for optimizing the parameters of structured nonlinear feedback controllers. The forward state and backward costate recursions are reformulated as root-finding problems and evaluated using a two-pass Gauss-Newton fixed-point method.

## Main contributions

- A policy-gradient formulation for discrete-time control-affine nonlinear systems, which is consistent with the deterministic policy gradient in previous work and recovers the standard LQR policy gradient as a special case.
- Develop a time-parallel policy-gradient method by parallelizing the evaluation of the state and costate trajectories.
- Stability-based conditions for horizon-uniform convergence guarantees.
- Parameter optimization evaluated on discrete-time LQR and the inertia-wheel pendulum.

## Inertia-wheel pendulum results

| Gradient descent | Projected gradient descent |
|:---:|:---:|
| ![Inertia-wheel pendulum with gradient descent](img/inertia_wheel_gd.png) | ![Inertia-wheel pendulum with projected gradient descent](img/inertia_wheel_pgd.gif) |

### Controller-parameter history

![History of the optimized controller parameters](img/parameter_history.png)

### Control-input comparison

The optimized controller reduces the large transient control effort relative to the baseline controller.

![Baseline and policy-gradient control inputs](img/control.png)

## Method

For a parameterized feedback controller $u_k=\pi(x_k,\theta)$, we minimize

$$
\mathcal{J}(x_0,\theta)= \sum_{k=0}^\infty \big(q(x_k) + \pi(x_k,\theta)^\top R \pi(x_k,\theta)\big).
$$

Each policy-gradient iteration:

1. evaluates the closed-loop state trajectory in parallel;
2. evaluates the corresponding costate trajectory in parallel;
3. combines the state and costate information to form the policy gradient; and
4. updates the controller parameters using projected gradient descent (PGD).

## Repository contents

| File | Description |
|---|---|
| `deer.py` | General Gauss-Newton trajectory solver. |
| `deer_LQR.py` | Implementation used for the LQR experiments. |
| `deer_constant_k_convergence.py` | Constant-gain LQR convergence experiment. |
| `inertia_wheel_pendulum.py` | Inertia-wheel pendulum model and baseline simulation. |
| `inertia_wheel_policy_optimization_deer.py` | Monte Carlo with gradient descent. |
| `inertia_wheel_projected_deer.py` | Monte Carlo with projected gradient descent. |
| `test_time.py` | Timing experiment for sequential and parallel trajectory evaluation. |

## Requirements

The experiments use Python 3 with JAX, NumPy, SciPy, Matplotlib, and Pillow:

```bash
pip install "jax[cpu]" numpy scipy matplotlib pillow
```

## Running the inertia-wheel experiments

Projected gradient descent:

```bash
python inertia_wheel_projected_deer.py
```

The scripts save plots, animations, optimized parameters, and NumPy result files in their respective result directories.

## Citation

Citation information will be added after publication.

## License

See [LICENSE](LICENSE).