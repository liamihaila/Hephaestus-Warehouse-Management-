# pip install "ray[rllib]" pettingzoo gymnasium pygame
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray import air, tune
from your_env_file import WarehouseTasksEnv  # path to the canvas file

def env_creator(_):
    return ParallelPettingZooEnv(
        WarehouseTasksEnv(coop_rewards=True, n_inbound_tasks=8, n_outbound_tasks=8)
    )

config = (
    PPOConfig()
    .environment(env=env_creator)
    .framework("torch")
    .rollouts(num_rollout_workers=2)
    .training(
        gamma=0.99, lr=3e-4, model={"vf_share_layers": True}
    )
    .multi_agent(
        policies={"shared": None},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared"
    )
)
algo = config.build()
for i in range(1000):
    result = algo.train()
    if i % 10 == 0:
        print(i, result["episode_reward_mean"])
