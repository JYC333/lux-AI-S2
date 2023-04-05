import copy
import warnings
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
from gym.wrappers import TimeLimit

from luxai_s2.state import StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import SB3Wrapper

from stable_baselines3.common import base_class
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)

from wrappers import (
    SimpleUnitDiscreteController,
    SimpleUnitObservationWrapper,
    ConvObservationWrapper,
)


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        own_factories = self.env.state.factories[agent]
        own_strain_id = self.env.state.teams[agent].factory_strains

        dis_to_factories = np.zeros(
            (self.env.state.env_cfg.map_size, self.env.state.env_cfg.map_size)
        )
        for i in range(len(dis_to_factories)):
            for j in range(len(dis_to_factories[i])):
                dis = float("inf")
                for k in own_factories.keys():
                    dis_k = np.sqrt(
                        (i - own_factories[k].pos.x) ** 2
                        + (j - own_factories[k].pos.y) ** 2
                    )
                    if dis_k < dis:
                        dis = dis_k
                dis_to_factories[i][j] = dis if dis != 0 else 1

        opp_factories = self.env.state.factories[opp_agent]
        opp_strain_id = self.env.state.teams[opp_agent].factory_strains
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["ore_dug"] = (
            stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        metrics["rubble"] = self.env.state.board.rubble

        metrics["lichen"] = self.env.state.board.lichen

        metrics["own_lichen_strains"] = np.zeros(
            (self.env.state.env_cfg.map_size, self.env.state.env_cfg.map_size)
        )
        metrics["opp_lichen_strains"] = np.zeros(
            (self.env.state.env_cfg.map_size, self.env.state.env_cfg.map_size)
        )
        for i in range(len(self.env.state.board.lichen_strains)):
            for j in range(len(self.env.state.board.lichen_strains[i])):
                if self.env.state.board.lichen_strains[i][j] in own_strain_id:
                    # print(
                    #     i,
                    #     j,
                    #     metrics["lichen"][i][j],
                    #     self.env.state.board.lichen_strains[i][j],
                    #     own_strain_id,
                    #     opp_strain_id,
                    # )
                    metrics["own_lichen_strains"][i][j] = 1
                if self.env.state.board.lichen_strains[i][j] in opp_strain_id:
                    # print(
                    #     i,
                    #     j,
                    #     self.env.state.board.lichen_strains[i][j],
                    #     metrics["lichen"][i][j],
                    #     own_strain_id,
                    #     opp_strain_id,
                    # )
                    metrics["opp_lichen_strains"][i][j] = 1
        # print(np.sum(metrics["own_lichen_strains"]), np.sum(metrics["lichen"]))

        metrics["power_consumption"] = (
            stats["consumption"]["power"]["LIGHT"]
            + stats["consumption"]["power"]["HEAVY"]
            + stats["consumption"]["power"]["FACTORY"]
        )

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
            rubble_dug_this_step = metrics["rubble"] - self.prev_step_metrics["rubble"]
            own_lichen_grow_this_step = (
                metrics["lichen"] * metrics["own_lichen_strains"]
                - self.prev_step_metrics["lichen"]
                * self.prev_step_metrics["own_lichen_strains"]
            )
            opp_lichen_grow_this_step = (
                metrics["lichen"] * metrics["opp_lichen_strains"]
                - self.prev_step_metrics["lichen"]
                * self.prev_step_metrics["opp_lichen_strains"]
            )
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            power_consumption_this_step = (
                metrics["power_consumption"]
                - self.prev_step_metrics["power_consumption"]
            )

            # we reward water production more as it is the most important resource for survival
            reward = (
                np.sum(rubble_dug_this_step / dis_to_factories) / 100
                + ore_dug_this_step / 200
                + ice_dug_this_step / 50
                + water_produced_this_step
                + np.sum(own_lichen_grow_this_step)
                - np.sum(opp_lichen_grow_this_step) / 1000
                # - power_consumption_this_step / 100000
            )
            # print(
            #     np.sum(rubble_dug_this_step * dis_to_factories),
            #     ore_dug_this_step,
            #     ice_dug_this_step,
            #     water_produced_this_step,
            #     np.sum(own_lichen_grow_this_step),
            # )

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def get_units_map(self):
        units = self.env.state.units["player_0"]
        units_map = np.zeros((48, 48))
        for unit in units:
            units_map[units[unit].pos.x][units[unit].pos.y] = 1
        return units_map

    def get_factories_map(self):
        factories = self.env.state.factories["player_0"]
        factories_map = np.zeros((48, 48))
        for factory in factories:
            factories_map[factories[factory].pos.x][factories[factory].pos.y] = 1
        return factories_map

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs


def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=100):
    def _init() -> gym.Env:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 2 for simplification and keeping returns consistent as we survive longer if there are more initial resources
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.

        env = SB3Wrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg),
        )
        # env = SimpleUnitObservationWrapper(
        #     env
        # )  # changes observation to include a few simple features
        env = ConvObservationWrapper(env)
        env = CustomEnvWrapper(env)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,
            np.array(env.env_method("get_units_map")),
            np.array(env.env_method("get_factories_map")),
            states,
        )
        observations, rewards, dones, infos = env.step(actions.detach().cpu())
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
