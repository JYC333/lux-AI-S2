import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces


# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()


class SimpleUnitDiscreteController(Controller):
    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.self_destruct_dims = 1
        self.recharge_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.self_destruct_dim_high = self.dig_dim_high + self.self_destruct_dims
        self.recharge_dim_high = self.self_destruct_dim_high + self.recharge_dims
        self.no_op_dim_high = self.recharge_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id, repeat):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, repeat, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id, amount, repeat):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, amount, repeat, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, amount, repeat):
        return np.array([2, 0, 4, amount, repeat, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, repeat):
        return np.array([3, 0, 0, 0, repeat, 1])

    def _is_self_destruct_action(self, id):
        return id < self.self_destruct_dim_high

    def _get_self_destruct_action(self):
        return np.array([4, 0, 0, 0, 0, 1])

    def _is_recharge_action(self, id):
        return id < self.self_destruct_dim_high

    def _get_recharge_action(self, amount, repeat):
        return np.array([5, 0, 0, amount, repeat, 1])

    def _get_robot_action_vec(self, choice, amount, repeat):
        if self._is_move_action(choice):
            action = self._get_move_action(choice, repeat)
        elif self._is_transfer_action(choice):
            action = self._get_transfer_action(choice, amount, repeat)
        elif self._is_pickup_action(choice):
            action = self._get_pickup_action(amount, repeat)
        elif self._is_dig_action(choice):
            action = self._get_dig_action(repeat)
        elif self._is_self_destruct_action(choice):
            action = self._get_self_destruct_action()
        elif self._is_recharge_action(choice):
            action = self._get_recharge_action(amount, repeat)
        else:
            # action is a no_op, so we don't update the action queue
            action = None

        return action

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        shared_obs = obs["player_0"]
        robots_actions = action[0]
        robots_action_amount = action[1:3]
        factories_actions = action[-1][0]
        lux_action = dict()
        units = shared_obs["units"][agent]
        for unit_id in units.keys():
            unit = units[unit_id]
            choices = robots_actions[:, unit["pos"][0], unit["pos"][1]]
            amounts, repeats = robots_action_amount[
                :, :, unit["pos"][0], unit["pos"][1]
            ]
            self.env_cfg.max_transfer_amount
            action_queue = []

            no_op = False
            for choice, amount, repeat in zip(choices, amounts, repeats):
                choice = choice.item()
                if 0 <= amount <= 1:
                    amount = amount.item()
                elif amount < 0:
                    amount = 0
                elif amount > 1:
                    amount = 1

                if 0 <= repeat <= 1:
                    repeat = repeat.item()
                elif repeat < 0:
                    repeat = 0
                elif repeat > 1:
                    repeat = 1

                action_step = self._get_robot_action_vec(
                    int(choice),
                    int(amount * self.env_cfg.max_transfer_amount) + 1,
                    int(repeat * 9999),
                )
                if action_step is not None:
                    action_queue.append(action_step)

            if len(action_queue) == 0:
                no_op = True

            # simple trick to help agents conserve power is to avoid updating the action queue
            # if the agent was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (
                    unit["action_queue"][0][:3] == action_queue[0][:3]
                ).all()
                if same_actions:
                    no_op = True
                # print(
                #     unit_id,
                #     same_actions,
                #     no_op,
                #     # unit["action_queue"][:2],
                #     # action_queue[:2],
                #     file=sys.stderr,
                # )
            if not no_op:
                lux_action[unit_id] = action_queue

        factories = shared_obs["factories"][agent]
        for factory_id in factories.keys():
            factory = factories[factory_id]
            factory_action = int(
                factories_actions[factory["pos"][0], factory["pos"][1]]
            )
            if factory_action == 3:
                continue
            else:
                lux_action[factory_id] = factory_action

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any], step: int):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        units = shared_obs["units"][agent]

        opp_strain_id = []
        factories_action_mask = np.zeros((48, 48, 4), dtype=bool)
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                if player != agent:
                    opp_strain_id.append(f_data["strain_id"])
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

                action_mask_vec = np.ones(4, dtype=bool)

                # build robot is valid only no own robot is on factory position
                for unit_id in units.keys():
                    if (
                        units[unit_id]["pos"][0] == f_pos[0]
                        and units[unit_id]["pos"][1] == f_pos[1]
                    ):
                        action_mask_vec[0] = False
                        action_mask_vec[1] = False

                # water lichen is valid only the remain water is enough to survive
                if player == agent:
                    owned_lichen_tiles = (
                        shared_obs["board"]["lichen_strains"] == f_data["strain_id"]
                    ).sum()
                    water_cost = np.ceil(
                        owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR
                    )

                    if water_cost < self.env_cfg.max_episode_length:
                        action_mask_vec[2] = False

                factories_action_mask[f_pos[0], f_pos[1]] = action_mask_vec

        robots_action_mask = np.zeros((48, 48, self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask_vec = np.zeros(self.total_act_dims, dtype=bool)
            # movement is always valid
            action_mask_vec[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask_vec[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask_vec[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask_vec[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask_vec[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # self destruction is valid only if on opponent's lichen
            if shared_obs["board"]["lichen_strains"][pos[0], pos[1]] in opp_strain_id:
                action_mask_vec[
                    self.self_destruct_dim_high
                    - self.self_destruct_dims : self.self_destruct_dim_high
                ] = True

            # recharge is always valid
            action_mask_vec[
                self.recharge_dim_high - self.recharge_dims : self.recharge_dim_high
            ] = True

            # no-op is always valid
            action_mask_vec[-1] = True

            robots_action_mask[pos[0], pos[1]] = action_mask_vec

        return np.concatenate((robots_action_mask, factories_action_mask), axis=2)
