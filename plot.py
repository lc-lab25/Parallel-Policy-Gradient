import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
baseline_file = "inertia_wheel_two_pass_deer_mc_results/training_results.npz"
policy_gradient_file = "inertia_wheel_projected_gradient_two_pass_deer_mc_results/training_results.npz"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
baseline_data = np.load(baseline_file)
pg_data = np.load(policy_gradient_file)

# Check the available keys if needed
print("Baseline keys:", baseline_data.files)
print("Policy-gradient keys:", pg_data.files)

# If the keys are stored without ".npy"
u_baseline = baseline_data["final_controls"]
t_baseline = baseline_data["time"]

u_pg = pg_data["final_controls"]
t_pg = pg_data["time"]

# Remove unnecessary singleton dimensions
u_baseline = np.squeeze(u_baseline)
t_baseline = np.squeeze(t_baseline)

u_pg = np.squeeze(u_pg)
t_pg = np.squeeze(t_pg)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(5, 3))

plt.plot(
    t_baseline[:1000],
    u_baseline,
    linewidth=2,
    # linestyle="--",
    color="blue",
    label="Baseline"
)

# plt.plot(
#     # t_baseline[:1001],
#     u_pg[:,1],
#     linewidth=2,
#     # linestyle="--",
#     color="orange",
#     label="$a_2$"
# )

# plt.plot(
#     # t_baseline[:1001],
#     u_pg[:,2],
#     linewidth=2,
#     # linestyle="--",
#     color="green",
#     label="$a_3$"
# )

# plt.plot(
#     # t_baseline[:1001],
#     u_pg[:,3],
#     linewidth=2,
#     # linestyle="--",
#     color="red",
#     label="$k_p$"
# )

# plt.plot(
#     # t_baseline[:1001],
#     u_pg[:,4],
#     linewidth=2,
#     # linestyle="--",
#     color="purple",
#     label="$k_v$"
# )

plt.plot(
    t_baseline[:1000],
    u_pg,
    linewidth=2,
    linestyle="--",
    color="red",
    label="Policy gradient"
)

# plt.plot(
#     # t_pg[:1000],
#     u_pg[:,0],
#     linewidth=2,
#     # linestyle="--",
#     color="blue",
#     label="Policy Gradient"
# )
# plt.xticks(np.arange(0, 80001, 20000))
plt.xlabel("Time (s)")
plt.ylabel("Control Input")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("parameter_history.png", dpi=800)
plt.close()
plt.show()