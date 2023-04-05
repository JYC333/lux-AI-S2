from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation


class ConvObservationWrapper(gym.ObservationWrapper):
    """ """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-1, 200, shape=(32, 48, 48))

    def can_water(self, factory):
        owned_lichen_tiles = (
            self.env.state.board.lichen_strains == factory.strain_id
        ).sum()
        water_cost = np.ceil(
            owned_lichen_tiles / self.env.state.env_cfg.LICHEN_WATERING_COST_FACTOR
        )
        return factory.cargo.water >= water_cost

    def observation(self, obs):
        return ConvObservationWrapper.convert_obs(
            obs, self.env.state.env_cfg, self.env.state.env_steps
        )

    def obs_mask(self):
        obs = self.env.reset()
        mask = np.zeros((self.env.action_space.n, 48, 48))
        units = obs["player_0"]["units"]["player_0"]
        for unit in units:
            mask[:, units[unit]["pos"][0], units[unit]["pos"][1]] = 1
        return mask

    @staticmethod
    def convert_obs(
        obs: Dict[str, Any], env_cfg: Any, step: int
    ) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        lichen_strains_map = shared_obs["board"]["lichen_strains"]

        for agent in obs.keys():
            obs_vec = [
                step // 50,
                step % 50,
                step % 50 // 30,
            ]

            own_light_robot_num = 0
            own_heavy_robot_num = 0
            opp_light_robot_num = 0
            opp_heavy_robot_num = 0

            own_total_ice = 0
            opp_total_ice = 0
            own_total_ore = 0
            opp_total_ore = 0
            own_total_water = 0
            opp_total_water = 0
            own_total_metal = 0
            opp_total_metal = 0
            own_total_power = 0
            opp_total_power = 0
            # own_total_lichen = 0
            # opp_total_lichen = 0
            own_total_water_cost = 0
            opp_total_water_cost = 0
            if_own_team_can_collect_ice = False
            if_opp_team_can_collect_ice = False
            if_own_team_can_grow_lichen = False
            if_opp_team_can_grow_lichen = False

            robot_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            own_light_robot_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            own_heavy_robot_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            opp_light_robot_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            opp_heavy_robot_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            light_robot_ice_amount_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            heavy_robot_ice_amount_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            light_robot_ore_amount_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            heavy_robot_ore_amount_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            light_robot_water_amount_map = np.zeros(
                (env_cfg.map_size, env_cfg.map_size)
            )
            heavy_robot_water_amount_map = np.zeros(
                (env_cfg.map_size, env_cfg.map_size)
            )
            light_robot_metal_amount_map = np.zeros(
                (env_cfg.map_size, env_cfg.map_size)
            )
            heavy_robot_metal_amount_map = np.zeros(
                (env_cfg.map_size, env_cfg.map_size)
            )
            robot_reach_cargo_limit_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            light_robot_power_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            heavy_robot_power_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            robot_reach_power_limit_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            for player in shared_obs["units"]:
                # print(shared_obs["units"][player], player)
                for unit in shared_obs["units"][player]:
                    pos = shared_obs["units"][player][unit]["pos"]
                    ice = shared_obs["units"][player][unit]["cargo"]["ice"]
                    ore = shared_obs["units"][player][unit]["cargo"]["ore"]
                    water = shared_obs["units"][player][unit]["cargo"]["water"]
                    metal = shared_obs["units"][player][unit]["cargo"]["metal"]
                    power = shared_obs["units"][player][unit]["power"]
                    robot_map[pos[0]][pos[1]] = 1
                    if player == "player_0":
                        own_total_ice += ice
                        own_total_ore += ore
                        own_total_water += water
                        own_total_metal += metal
                        own_total_power += power
                        if shared_obs["units"][player][unit]["unit_type"] == "LIGHT":
                            own_light_robot_map[pos[0]][pos[1]] = 1
                            own_light_robot_num += 1
                            light_robot_ice_amount_map[pos[0]][pos[1]] = ice
                            light_robot_ore_amount_map[pos[0]][pos[1]] = ore
                            light_robot_water_amount_map[pos[0]][pos[1]] = water
                            light_robot_metal_amount_map[pos[0]][pos[1]] = metal
                            light_robot_power_map[pos[0]][pos[1]] = power

                            if (
                                ice_map[pos[0]][pos[1]]
                                and power > env_cfg.ROBOTS["LIGHT"].DIG_COST
                            ):
                                if_own_team_can_collect_ice |= True

                            if (
                                ice + ore + water + metal
                                >= env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
                            ):
                                robot_reach_cargo_limit_map[pos[0]][pos[1]] = 1

                            if power >= env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY:
                                robot_reach_power_limit_map[pos[0]][pos[1]] = 1
                        else:
                            own_heavy_robot_map[pos[0]][pos[1]] = 1
                            own_heavy_robot_num += 1
                            heavy_robot_ice_amount_map[pos[0]][pos[1]] = ice
                            heavy_robot_ore_amount_map[pos[0]][pos[1]] = ore
                            heavy_robot_water_amount_map[pos[0]][pos[1]] = water
                            heavy_robot_metal_amount_map[pos[0]][pos[1]] = metal
                            heavy_robot_power_map[pos[0]][pos[1]] = power

                            if (
                                ice_map[pos[0]][pos[1]]
                                and power > env_cfg.ROBOTS["HEAVY"].DIG_COST
                            ):
                                if_own_team_can_collect_ice |= True

                            if (
                                ice + ore + water + metal
                                >= env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
                            ):
                                robot_reach_cargo_limit_map[pos[0]][pos[1]] = 1

                            if power >= env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY:
                                robot_reach_power_limit_map[pos[0]][pos[1]] = 1
                    else:
                        opp_total_ice += ice
                        opp_total_ore += ore
                        opp_total_water += water
                        opp_total_metal += metal
                        opp_total_power += power
                        if shared_obs["units"][player][unit]["unit_type"] == "LIGHT":
                            opp_light_robot_map[pos[0]][pos[1]] = 1
                            opp_light_robot_num += 1
                            light_robot_ice_amount_map[pos[0]][pos[1]] = ice
                            light_robot_ore_amount_map[pos[0]][pos[1]] = ore
                            light_robot_water_amount_map[pos[0]][pos[1]] = water
                            light_robot_metal_amount_map[pos[0]][pos[1]] = metal
                            light_robot_power_map[pos[0]][pos[1]] = power

                            if (
                                ice_map[pos[0]][pos[1]]
                                and power > env_cfg.ROBOTS["LIGHT"].DIG_COST
                            ):
                                if_opp_team_can_collect_ice |= True

                            if (
                                ice + ore + water + metal
                                >= env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
                            ):
                                robot_reach_cargo_limit_map[pos[0]][pos[1]] = 1

                            if power >= env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY:
                                robot_reach_power_limit_map[pos[0]][pos[1]] = 1
                        else:
                            opp_heavy_robot_map[pos[0]][pos[1]] = 1
                            opp_heavy_robot_num += 1
                            heavy_robot_ice_amount_map[pos[0]][pos[1]] = ice
                            heavy_robot_ore_amount_map[pos[0]][pos[1]] = ore
                            heavy_robot_water_amount_map[pos[0]][pos[1]] = water
                            heavy_robot_metal_amount_map[pos[0]][pos[1]] = metal
                            heavy_robot_power_map[pos[0]][pos[1]] = power

                            if (
                                ice_map[pos[0]][pos[1]]
                                and power > env_cfg.ROBOTS["HEAVY"].DIG_COST
                            ):
                                if_opp_team_can_collect_ice |= True

                            if (
                                ice + ore + water + metal
                                >= env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
                            ):
                                robot_reach_cargo_limit_map[pos[0]][pos[1]] = 1

                            if power >= env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY:
                                robot_reach_power_limit_map[pos[0]][pos[1]] = 1

            obs_mat = np.dstack(
                (
                    robot_map,
                    own_light_robot_map,
                    own_heavy_robot_map,
                    opp_light_robot_map,
                    opp_heavy_robot_map,
                ),
            )

            factory_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            own_factory_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            opp_factory_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            own_factory_strain_id = set()
            opp_factory_strain_id = set()
            factory_ice_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            factory_ore_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            factory_water_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            factory_metal_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            factory_power_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            for player in shared_obs["factories"]:
                for factory in shared_obs["factories"][player]:
                    pos = shared_obs["factories"][player][factory]["pos"]
                    strain_id = shared_obs["factories"][player][factory]["strain_id"]
                    factory_ice_map[pos[0]][pos[1]] = shared_obs["factories"][player][
                        factory
                    ]["cargo"]["ice"]
                    factory_ore_map[pos[0]][pos[1]] = shared_obs["factories"][player][
                        factory
                    ]["cargo"]["ore"]
                    factory_water_map[pos[0]][pos[1]] = shared_obs["factories"][player][
                        factory
                    ]["cargo"]["water"]
                    factory_metal_map[pos[0]][pos[1]] = shared_obs["factories"][player][
                        factory
                    ]["cargo"]["metal"]
                    factory_power_map[pos[0]][pos[1]] = shared_obs["factories"][player][
                        factory
                    ]["power"]

                    water_cost = np.ceil(
                        (shared_obs["board"]["lichen_strains"] == strain_id).sum()
                        / env_cfg.LICHEN_WATERING_COST_FACTOR
                    )

                    # Factory is a 3*3 square
                    factory_map[pos[0]][pos[1]] = 1
                    factory_map[pos[0]][pos[1] - 1] = 1
                    factory_map[pos[0]][pos[1] + 1] = 1
                    factory_map[pos[0] - 1][pos[1]] = 1
                    factory_map[pos[0] - 1][pos[1] - 1] = 1
                    factory_map[pos[0] - 1][pos[1] + 1] = 1
                    factory_map[pos[0] + 1][pos[1]] = 1
                    factory_map[pos[0] + 1][pos[1] - 1] = 1
                    factory_map[pos[0] + 1][pos[1] + 1] = 1

                    if player == "player_0":
                        own_factory_strain_id.add(strain_id)
                        own_total_water_cost += water_cost + 1
                        if_own_team_can_grow_lichen |= (
                            shared_obs["factories"][player][factory]["cargo"]["water"]
                            >= water_cost
                        )

                        own_factory_map[pos[0]][pos[1]] = 1
                        own_factory_map[pos[0]][pos[1] - 1] = 1
                        own_factory_map[pos[0]][pos[1] + 1] = 1
                        own_factory_map[pos[0] - 1][pos[1]] = 1
                        own_factory_map[pos[0] - 1][pos[1] - 1] = 1
                        own_factory_map[pos[0] - 1][pos[1] + 1] = 1
                        own_factory_map[pos[0] + 1][pos[1]] = 1
                        own_factory_map[pos[0] + 1][pos[1] - 1] = 1
                        own_factory_map[pos[0] + 1][pos[1] + 1] = 1
                    else:
                        opp_factory_strain_id.add(strain_id)
                        opp_total_water_cost += water_cost + 1
                        if_opp_team_can_grow_lichen |= (
                            shared_obs["factories"][player][factory]["cargo"]["water"]
                            >= water_cost
                        )

                        opp_factory_map[pos[0]][pos[1]] = 1
                        opp_factory_map[pos[0]][pos[1] - 1] = 1
                        opp_factory_map[pos[0]][pos[1] + 1] = 1
                        opp_factory_map[pos[0] - 1][pos[1]] = 1
                        opp_factory_map[pos[0] - 1][pos[1] - 1] = 1
                        opp_factory_map[pos[0] - 1][pos[1] + 1] = 1
                        opp_factory_map[pos[0] + 1][pos[1]] = 1
                        opp_factory_map[pos[0] + 1][pos[1] - 1] = 1
                        opp_factory_map[pos[0] + 1][pos[1] + 1] = 1

            obs_mat = np.dstack(
                (obs_mat, factory_map, own_factory_map, opp_factory_map)
            )

            obs_mat = np.dstack(
                (
                    obs_mat,
                    ice_map,
                    shared_obs["board"]["ore"],
                    shared_obs["board"]["rubble"] / 100,
                )
            )

            own_lichen_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            opp_lichen_map = np.zeros((env_cfg.map_size, env_cfg.map_size))
            for strain_id in own_factory_strain_id:
                own_lichen_map += np.where(lichen_strains_map == strain_id, 1, 0)
            for strain_id in opp_factory_strain_id:
                opp_lichen_map += np.where(lichen_strains_map == strain_id, 1, 0)

            own_total_lichen = np.sum(own_lichen_map * shared_obs["board"]["lichen"])
            opp_total_lichen = np.sum(opp_lichen_map * shared_obs["board"]["lichen"])

            obs_mat = np.dstack(
                (
                    obs_mat,
                    shared_obs["board"]["lichen"] / 100,
                    own_lichen_map,
                    opp_lichen_map,
                )
            )

            obs_mat = np.dstack(
                (
                    obs_mat,
                    light_robot_ice_amount_map / 100,
                    heavy_robot_ice_amount_map / 1000,
                    light_robot_ore_amount_map / 100,
                    heavy_robot_ore_amount_map / 1000,
                    light_robot_water_amount_map / 100,
                    heavy_robot_water_amount_map / 1000,
                    light_robot_metal_amount_map / 100,
                    heavy_robot_metal_amount_map / 1000,
                    robot_reach_cargo_limit_map,
                    light_robot_power_map / 150,
                    heavy_robot_power_map / 3000,
                    robot_reach_power_limit_map,
                )
            )

            obs_mat = np.dstack(
                (
                    obs_mat,
                    factory_ice_map / 1000,
                    factory_ore_map / 1000,
                    factory_water_map / 1000,
                    factory_metal_map / 1000,
                    factory_power_map / 3000,
                )
            )

            obs_vec += [
                own_light_robot_num / 30,
                own_heavy_robot_num / 10,
                opp_light_robot_num / 30,
                opp_heavy_robot_num / 10,
                own_total_ice / 1000 / shared_obs["board"]["factories_per_team"],
                opp_total_ice / 1000 / shared_obs["board"]["factories_per_team"],
                own_total_ore / 1000 / shared_obs["board"]["factories_per_team"],
                opp_total_ore / 1000 / shared_obs["board"]["factories_per_team"],
                own_total_water / 1000 / shared_obs["board"]["factories_per_team"],
                opp_total_water / 1000 / shared_obs["board"]["factories_per_team"],
                own_total_metal / 1000 / shared_obs["board"]["factories_per_team"],
                opp_total_metal / 1000 / shared_obs["board"]["factories_per_team"],
                own_total_power / 4000 / shared_obs["board"]["factories_per_team"],
                opp_total_power / 4000 / shared_obs["board"]["factories_per_team"],
                own_total_lichen / 1000 / shared_obs["board"]["factories_per_team"],
                opp_total_lichen / 1000 / shared_obs["board"]["factories_per_team"],
                own_total_water_cost / 100 / shared_obs["board"]["factories_per_team"],
                opp_total_water_cost / 100 / shared_obs["board"]["factories_per_team"],
                1 if if_own_team_can_collect_ice else 0,
                1 if if_opp_team_can_collect_ice else 0,
                1 if if_own_team_can_grow_lichen else 0,
                1 if if_opp_team_can_grow_lichen else 0,
            ]

            obs_vec += [-1] * (48 * 48 - len(obs_vec))
            obs_vec = np.array(obs_vec).reshape((env_cfg.map_size, env_cfg.map_size))

            obs_mat = np.dstack((obs_vec, obs_mat))

            obs_mat = obs_mat.reshape((-1, env_cfg.map_size, env_cfg.map_size))

            observation[agent] = obs_mat

        return observation
