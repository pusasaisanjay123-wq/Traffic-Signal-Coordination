
import os, glob
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Sumo"
os.makedirs("results", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import supersuit as ss
from stable_baselines3 import DQN

NET_FILE = "nets/2x2grid/2x2.net.xml"
ROUTE_FILE = "nets/2x2grid/2x2.rou.xml"

# Helper to create Multi-Agent Env with UNSEEN Traffic (seed 999)
def make_unseen_env(csv_name, seed=999):
    env = sumo_rl.parallel_env(
        net_file=NET_FILE, 
        route_file=ROUTE_FILE, 
        out_csv_name=csv_name,
        use_gui=False,
        num_seconds=3600,
        additional_sumo_cmd=f"--seed {seed}"  # <--- Fixed seed argument!
    )
    env.render_mode = None
    if hasattr(env, "unwrapped"):
        env.unwrapped.render_mode = None

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env

print("--- Testing Multi-Agent Model on Unseen Traffic (Seed 999) ---")

# 1. Run Baseline on Unseen Traffic
print("1. Running Baseline...")
env_b = make_unseen_env("results/unseen_marl_base", seed=999)
obs = env_b.reset()
for _ in range(720):
    action = [env_b.action_space.sample() for _ in range(env_b.num_envs)]
    obs, _, _, _ = env_b.step(action)
env_b.close()

# 2. Run Trained MARL Agent on Unseen Traffic
print("2. Running Trained MARL...")
env_d = make_unseen_env("results/unseen_marl_dqn", seed=999)
model = DQN.load("results/marl_model")
obs = env_d.reset()
for _ in range(720):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env_d.step(action)
env_d.close()

# 3. Plot Comparison
print("3. Generating Graph...")
b_files = sorted(glob.glob("results/unseen_marl_base*.csv"))
d_files = sorted(glob.glob("results/unseen_marl_dqn*.csv"))

b_df = pd.read_csv(b_files[-1]).groupby('step').mean(numeric_only=True).reset_index()
d_df = pd.read_csv(d_files[-1]).groupby('step').mean(numeric_only=True).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(b_df["step"], b_df["system_mean_waiting_time"], label="Fixed Baseline", color="red")
axes[0].plot(d_df["step"], d_df["system_mean_waiting_time"], label="Trained MARL (Unseen)", color="green")
axes[0].set_title("Unseen Traffic: Avg Wait Time (s)")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(b_df["step"], b_df["system_mean_speed"], label="Fixed Baseline", color="red")
axes[1].plot(d_df["step"], d_df["system_mean_speed"], label="Trained MARL (Unseen)", color="green")
axes[1].set_title("Unseen Traffic: Avg Speed (m/s)")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/marl_overfitting_test.png", dpi=300)
plt.show()
print("SUCCESS: Saved to results/marl_overfitting_test.png!")