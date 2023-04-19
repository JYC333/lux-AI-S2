import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import evaluate_policy
from models.ResConv import ResNet, ActorOutput, CriticOutput


class ConvNetwork(nn.Module):
    def __init__(self, env=None, env_cfg=None, cuda=True):
        super(ConvNetwork, self).__init__()
        if env:
            self.queue_size = env.get_attr("state")[0].env_cfg.UNIT_ACTION_QUEUE_SIZE
            self.map_size = env.get_attr("state")[0].env_cfg.map_size

        if env_cfg:
            self.queue_size = env_cfg.UNIT_ACTION_QUEUE_SIZE
            self.map_size = env_cfg.map_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda else "cpu"
        )

        base_res_net = ResNet(
            encoding_out_channels=16,
            res_in_channels=64,
            res_out_channels=128,
            n_one_hots=3,
            n_others=22,
            n_map_features=31,
            one_hot_num_classes=[200, 50, 2],
            h=self.map_size,
            w=self.map_size,
        )

        self.critic = nn.Sequential(
            base_res_net,
            CriticOutput(res_out_channels=128),
        )
        self.actor = nn.Sequential(
            base_res_net,
            ActorOutput(
                res_out_channels=128,
                n_robots_action=14,
                n_factories_action=4,
            ),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.map_size))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, units_map, factories_map, actions=None):
        logits_robots, logits_robots_amount, logits_factories = self.actor(x)

        logits_robots = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots, units_map)
            ],
            0,
        )
        logits_robots_amount = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots_amount, units_map)
            ],
            0,
        )
        logits_factories = torch.cat(
            [
                (logits * factories).unsqueeze(0)
                for logits, factories in zip(logits_factories, factories_map)
            ],
            0,
        )

        # Multidiscrete robots actions
        split_robots_logits = torch.split(
            logits_robots.transpose(1, -1).transpose(1, 2), 1, dim=1
        )
        multi_robots_probs = [
            Categorical(logits=logits) for logits in split_robots_logits
        ]

        if actions is None:
            robots_actions = torch.cat(
                [probs.sample() for probs in multi_robots_probs], 1
            )
        else:
            robots_actions = actions[:, 0]

        robots_log_prob = torch.cat(
            [probs.log_prob(a) for a, probs in zip(robots_actions, multi_robots_probs)],
            1,
        )
        robots_entropy = torch.cat([probs.entropy() for probs in multi_robots_probs], 1)
        robots_actions = robots_actions.unsqueeze(1)

        # Multicontinuous robots actions
        split_robots_amount_logits = torch.split(logits_robots_amount, 1, dim=2)
        split_robots_amount_logstd = torch.split(
            self.actor_logstd.expand_as(logits_robots_amount),
            1,
            dim=2,
        )
        split_robots_amount_std = [
            torch.exp(logstd) for logstd in split_robots_amount_logstd
        ]
        multi_robots_amount_probs = [
            Normal(logits, logstd)
            for logits, logstd in zip(
                split_robots_amount_logits, split_robots_amount_std
            )
        ]

        if actions is None:
            robots_amount_actions = torch.cat(
                [probs.sample() for probs in multi_robots_amount_probs],
                2,
            )
        else:
            robots_amount_actions = actions[:, 1:3]

        robots_amount_log_prob = torch.cat(
            [
                probs.log_prob(a)
                for a, probs in zip(robots_amount_actions, multi_robots_amount_probs)
            ],
            2,
        )
        robots_amount_entropy = torch.cat(
            [probs.entropy() for probs in multi_robots_amount_probs],
            2,
        )

        # Discrete factories actions
        factories_probs = Categorical(logits=logits_factories.transpose(1, -1))
        if actions is None:
            factories_actions = factories_probs.sample()
        else:
            factories_actions = actions[:, 3, 0]
        factories_log_prob = factories_probs.log_prob(factories_actions)
        factories_entropy = factories_probs.entropy()
        factories_actions = factories_actions.unsqueeze(1).unsqueeze(1)
        factories_actions = factories_actions.expand(
            (-1, -1, self.queue_size, self.map_size, self.map_size)
        )

        if actions is None:
            actions = torch.cat(
                [robots_actions, robots_amount_actions, factories_actions], 1
            )

        log_probs = (
            robots_log_prob.sum()
            + robots_amount_log_prob.sum()
            + factories_log_prob.sum()
        )
        entropys = (
            robots_entropy.sum() + robots_amount_entropy.sum() + factories_entropy.sum()
        )
        return actions, log_probs, entropys, self.critic(x)

    def predict_action(self, x, units_map, factories_map, action_mask=None):
        x = x.to(self.device)
        units_map = torch.Tensor(units_map).to(self.device).unsqueeze(0)
        factories_map = torch.Tensor(factories_map).to(self.device).unsqueeze(0)
        logits_robots, logits_robots_amount, logits_factories = self.actor(x)
        if not action_mask is None:
            robots_action_mask = action_mask[:, :, :, :-4].unsqueeze(0)
            factories_action_mask = action_mask[:, :, :, -4:]

        logits_robots = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots, units_map)
            ],
            0,
        )
        logits_robots_amount = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots_amount, units_map)
            ],
            0,
        )
        logits_factories = torch.cat(
            [
                (logits * factories).unsqueeze(0)
                for logits, factories in zip(logits_factories, factories_map)
            ],
            0,
        )

        # Multidiscrete robots actions
        split_robots_logits = torch.split(
            logits_robots.transpose(1, -1).transpose(1, 2), 1, dim=1
        )
        if not action_mask is None:
            split_robots_logits[0][~robots_action_mask] = -1e8

        multi_robots_probs = [
            Categorical(logits=logits) for logits in split_robots_logits
        ]
        robots_actions = torch.cat([probs.sample() for probs in multi_robots_probs], 1)
        robots_actions = robots_actions.unsqueeze(1)

        # Multicontinuous robots actions
        split_robots_amount_logits = torch.split(logits_robots_amount, 1, dim=2)
        split_robots_amount_logstd = torch.split(
            self.actor_logstd.expand_as(logits_robots_amount),
            1,
            dim=2,
        )
        split_robots_amount_std = [
            torch.exp(logstd) for logstd in split_robots_amount_logstd
        ]
        multi_robots_amount_probs = [
            Normal(logits, logstd)
            for logits, logstd in zip(
                split_robots_amount_logits, split_robots_amount_std
            )
        ]

        robots_amount_actions = torch.cat(
            [probs.sample() for probs in multi_robots_amount_probs],
            2,
        )

        # Discrete factories actions
        logits_factories = logits_factories.transpose(1, -1)
        if not action_mask is None:
            logits_factories[~factories_action_mask] = -1e8
        factories_probs = Categorical(logits=logits_factories)
        factories_actions = factories_probs.sample()
        factories_actions = factories_actions.unsqueeze(1).unsqueeze(1)
        factories_actions = factories_actions.expand(
            (-1, -1, self.queue_size, self.map_size, self.map_size)
        )

        actions = torch.cat(
            [robots_actions, robots_amount_actions, factories_actions], 1
        )

        return actions[0]

    def predict(self, obs, units_map, factories_map, states=None):
        obs = torch.Tensor(obs).to(self.device)
        units_map = torch.Tensor(units_map).to(self.device)
        factories_map = torch.Tensor(factories_map).to(self.device)
        logits_robots, logits_robots_amount, logits_factories = self.actor(obs)

        logits_robots = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots, units_map)
            ],
            0,
        )
        logits_robots_amount = torch.cat(
            [
                (logits * units).unsqueeze(0)
                for logits, units in zip(logits_robots_amount, units_map)
            ],
            0,
        )
        logits_factories = torch.cat(
            [
                (logits * factories).unsqueeze(0)
                for logits, factories in zip(logits_factories, factories_map)
            ],
            0,
        )

        # Multidiscrete robots actions
        split_robots_logits = torch.split(
            logits_robots.transpose(1, -1).transpose(1, 2), 1, dim=1
        )
        multi_robots_probs = [
            Categorical(logits=logits) for logits in split_robots_logits
        ]
        robots_actions = torch.cat([probs.sample() for probs in multi_robots_probs], 1)
        robots_actions = robots_actions.unsqueeze(1)

        # Multicontinuous robots actions
        split_robots_amount_logits = torch.split(logits_robots_amount, 1, dim=2)
        split_robots_amount_logstd = torch.split(
            self.actor_logstd.expand_as(logits_robots_amount),
            1,
            dim=2,
        )
        split_robots_amount_std = [
            torch.exp(logstd) for logstd in split_robots_amount_logstd
        ]
        multi_robots_amount_probs = [
            Normal(logits, logstd)
            for logits, logstd in zip(
                split_robots_amount_logits, split_robots_amount_std
            )
        ]

        robots_amount_actions = torch.cat(
            [probs.sample() for probs in multi_robots_amount_probs],
            2,
        )

        # Discrete factories actions
        factories_probs = Categorical(logits=logits_factories.transpose(1, -1))
        factories_actions = factories_probs.sample()
        factories_actions = factories_actions.unsqueeze(1).unsqueeze(1)
        factories_actions = factories_actions.expand(
            (-1, -1, self.queue_size, self.map_size, self.map_size)
        )

        actions = torch.cat(
            [robots_actions, robots_amount_actions, factories_actions], 1
        )

        return actions, states


class PPO:
    def __init__(
        self,
        env,
        eval_env=None,
        learning_rate=2.5e-4,
        total_timesteps=1000000,
        eval_freq=4000,
        num_envs=4,
        num_steps=128,
        anneal_lr=True,
        gae=True,
        gamma=True,
        gae_lambda=True,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=True,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        # Other setting
        seed=1,
        cuda=True,
        torch_deterministic=True,
        resume=False,
        model_path=None,
        local_rank=None,
    ) -> None:
        self.learning_rate = learning_rate
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.seed = seed
        self.loacl_rank = local_rank

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        torch.distributed.init_process_group(backend="nccl")

        # env setup
        self.env = env
        self.eval_env = eval_env
        self.best_model = float("-inf")

        if resume:
            print("Resume training")
            self.agent, checkpoint = PPO.load(
                model_path, resume=True, local_rank=local_rank, env=env
            )
            self.agent = self.agent.to(self.device)
            self.optimizer = optim.Adam(
                self.agent.parameters(), lr=self.learning_rate, eps=1e-5
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            if model_path:
                print(f"Using model: {model_path}")
                self.agent = PPO.load(model_path, env=env).to(self.device)
            else:
                print("Creating new model")
                self.agent = ConvNetwork(env=env).to(self.device)
            self.optimizer = optim.Adam(
                self.agent.parameters(), lr=self.learning_rate, eps=1e-5
            )

        self.agent = torch.nn.parallel.DistributedDataParallel(
            self.agent,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

        out = evaluate_policy(self.agent.module, self.eval_env, deterministic=False)
        self.best_model = out[0] - out[1]
        print(f"{local_rank}-Model Score before training:{out[0] - out[1]}, ({out})")

    def train(self):
        if self.loacl_rank == 0:
            run_name = f"LUX__{self.seed}__{int(time.time())}"
            writer = SummaryWriter(f"runs/{run_name}")

        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs)
            + (4, 20)
            + self.env.observation_space.shape[1:]
        ).to(self.device)
        units_map = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.observation_space.shape[1:]
        ).to(self.device)
        factories_map = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.observation_space.shape[1:]
        ).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size
        eval_update = self.eval_freq // self.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    units_map[step] = torch.Tensor(
                        np.array(self.env.env_method("get_units_map"))
                    ).to(self.device)
                    factories_map[step] = torch.Tensor(
                        np.array(self.env.env_method("get_factories_map"))
                    ).to(self.device)

                    action, logprob, _, value = self.agent.module.get_action_and_value(
                        next_obs, units_map[step], factories_map[step]
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        print(
                            f"{self.loacl_rank}-global_step={global_step}, episodic_return={item['episode']['r']}"
                        )
                        if self.loacl_rank == 0:
                            writer.add_scalar(
                                "charts/episodic_return",
                                item["episode"]["r"],
                                global_step,
                            )
                            writer.add_scalar(
                                "charts/episodic_length",
                                item["episode"]["l"],
                                global_step,
                            )
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.module.get_value(next_obs).reshape(1, -1)
                if self.gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(
                (-1,) + (4, 20) + self.env.observation_space.shape[1:]
            )
            b_units_map = units_map.reshape(
                (-1,) + self.env.observation_space.shape[1:]
            )
            b_factories_map = factories_map.reshape(
                (-1,) + self.env.observation_space.shape[1:]
            )
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                    ) = self.agent.module.get_action_and_value(
                        b_obs[mb_inds],
                        b_units_map[mb_inds],
                        b_factories_map[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if update % eval_update == 0 and self.eval_env:
                out = evaluate_policy(
                    self.agent.module, self.eval_env, deterministic=False
                )
                checkpoint = {
                    "net": self.agent.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                self.save(f"checkpoint-{self.loacl_rank}", checkpoint=checkpoint)
                self.save(f"latest_model-{self.loacl_rank}")
                print(f"{self.loacl_rank}-Evalutate Score:{out[0] - out[1]}, ({out})")
                if out[0] - out[1] > self.best_model:
                    self.best_model = out[0] - out[1]
                    print(f"Saving best model: {self.best_model}")
                    self.save(f"best_model-{self.loacl_rank}")

            if self.loacl_rank == 0:
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar(
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar(
                    "losses/explained_variance", explained_var, global_step
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

        self.env.close()
        if self.loacl_rank == 0:
            writer.close()

    def save(self, path, checkpoint=None):
        if checkpoint is None:
            torch.save(self.agent.module.state_dict(), path)
        else:
            torch.save(checkpoint, path)

    @staticmethod
    def load(path, resume=False, local_rank="0", **kwargs):
        model = ConvNetwork(**kwargs)

        if resume:
            checkpoint = torch.load(path + "-" + str(local_rank))
            model.load_state_dict(checkpoint["net"])
            model.eval()
            return model, checkpoint
        else:
            model.load_state_dict(torch.load(path))
            model.eval()
            return model


if __name__ == "__main__":
    model = PPO()
    model.train()
