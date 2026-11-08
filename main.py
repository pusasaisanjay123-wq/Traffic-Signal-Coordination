
import os
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
os.makedirs("results", exist_ok=True)

import glob, shutil
import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import supersuit as ss
from stable_baselines3 import DQN

# Setup nets
try:
    src = os.path.join(os.path.dirname(sumo_rl.__file__), 'nets')
    shutil.copytree(src, 'nets', dirs_exist_ok=True)
except: pass

NET_FILE = "nets/2x2grid/2x2.net.xml"
ROUTE_FILE = "nets/2x2grid/2x2.rou.xml"

def make_env(csv_name, use_gui=False):
    env = sumo_rl.parallel_env(
        net_file=NET_FILE, route_file=ROUTE_FILE, out_csv_name=csv_name,
        use_gui=use_gui, num_seconds=3600
    )
    # Compatibility fix
    mode = "human" if use_gui else None
    env.render_mode = mode
    if hasattr(env, "unwrapped"):
        env.unwrapped.render_mode = mode
        
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env

# 1. Train Multi-Agent
print("--- 1. Training Multi-Agent DQN (2x2 Grid) ---")
env = make_env("results/marl_train", use_gui=False)
model = DQN("MlpPolicy", env, learning_rate=1e-3, verbose=0)
model.learn(total_timesteps=40000)
model.save("results/marl_model")
env.close()

# 2. Evaluate Multi-Agent
print("--- 2. Evaluating Multi-Agent DQN ---")
env = make_env("results/marl_eval", use_gui=True) # Set True to watch 4 lights
model = DQN.load("results/marl_model")
obs = env.reset()
for _ in range(720):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
env.close()

# 3. 6-Metric Plot
print("--- 3. Generating 6-Metric Plot ---")
train_files = sorted(glob.glob("results/marl_train_*.csv"), key=os.path.getmtime)
b_df = pd.read_csv(train_files[0]).groupby('step').mean(numeric_only=True).reset_index()
d_df = pd.read_csv(train_files[-1]).groupby('step').mean(numeric_only=True).reset_index()

# 6 Key Metrics
metrics = [
    ("system_mean_waiting_time", "1. Avg Wait Time (s)"),
    ("system_total_stopped", "2. Total Stopped Cars"),
    ("system_mean_speed", "3. Avg Speed (m/s)"),
    ("system_total_waiting_time", "4. Cumulative Delay (s)"),
    ("system_total_stopped", "5. System Queue Density"),
    ("agents_total_accumulated_waiting_time", "6. Multi-Agent Accumulated Wait")
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (col, label) in enumerate(metrics):
    if col in b_df.columns and col in d_df.columns:
        axes[i].plot(b_df["step"], b_df[col], label="Untrained (Ep 1)", color="red", alpha=0.7)
        axes[i].plot(d_df["step"], d_df[col], label="Trained MARL", color="green", alpha=0.8)
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel("Simulation Step")
    else:
        axes[i].text(0.5, 0.5, f"{label}\n(Data N/A)", ha='center', va='center')
        axes[i].set_title(label)

plt.suptitle("Multi-Agent RL (2x2 Grid): 6 Real-World Traffic Metrics", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("results/multi_agent_6metrics.png", dpi=300)
plt.show()
print("SUCCESS: Saved results/multi_agent_6metrics.png")
