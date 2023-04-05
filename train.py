"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""

import os.path as osp

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from models.ppo import PPO
from utils import evaluate_policy, make_env


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a heavy unit to dig ice and transfer it back to a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, will resume the training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="The model to load",
    )
    args = parser.parse_args()
    return args


def main(args):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"

    env = SubprocVecEnv(
        [
            make_env(env_id, i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    env.reset()

    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(5)]
    )
    eval_env.reset()

    model = PPO(
        env,
        eval_env,
        num_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        resume=args.resume,
        model_path=args.model_path,
    )

    if args.eval:
        out = evaluate_policy(
            PPO.load(args.model_path, env_cfg=eval_env),
            eval_env,
            render=False,
            deterministic=False,
        )
        print(out)
    else:
        model.train()


if __name__ == "__main__":
    # python ../examples/sb3.py -s 42 -n 1
    main(parse_args())
