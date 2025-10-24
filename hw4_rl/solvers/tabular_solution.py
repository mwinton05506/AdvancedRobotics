#!/usr/bin/env python
"""
CSCI 5302 HW2: Tabular Solution Implementation

This module implements various reinforcement learning algorithms for solving gridworld
problems using tabular methods. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- GridworldSolver: A class that handles policy computation and visualization
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from hw4_rl.envs import GridworldEnv

student_name = "My Name"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy implementation for discrete state/action spaces.

    This class implements a tabular policy and value function for reinforcement learning
    in discrete state/action spaces. It maintains mappings from states to values and
    from states to action probability distributions.

    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        state_ranges (np.ndarray): Array of [min, max) ranges for each state dimension
        _value_function (np.ndarray): Array mapping states to their values
        _policy (np.ndarray): Array mapping states to action probability distributions
    """

    def __init__(self, n_states: int, state_ranges: np.ndarray, n_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            n_states: Number of discrete states
            state_ranges: Array of [min, max) ranges for each state dimension
            n_actions: Number of possible actions
        """
        self.num_states = n_states
        self.num_actions = n_actions
        self.state_ranges = state_ranges

        # Create data structure to store mapping from state to value
        self._value_function = np.zeros(shape=n_states)

        # Create data structure to store array with probability of each action for each state
        self._policy = np.random.uniform(0, 1, size=(n_states, self.num_actions))

    def get_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Sample an action from the policy's action distribution for the given state.

        Args:
            state: The current state index or state coordinate vector

        Returns:
            The sampled action index
        """
        # Convert state to integer if it's a numpy array
        if isinstance(state, np.ndarray):
            state = self.get_state_index_from_coordinates(state)
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value for a given state.

        Args:
            state: The state index
            value: The value to set for this state
        """
        self._value_function[state] = value

    def get_state_value(self, state: Union[int, np.ndarray]) -> float:
        """
        Get the value for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            The value for the given state
        """
        if isinstance(state, int):
            return self._value_function[state]
        else:
            # Map state vector to state index
            return self._value_function[self.get_state_index_from_coordinates(state)]

    def get_state_index_from_coordinates(self, state: np.ndarray) -> int:
        """
        Convert a state coordinate vector to its corresponding state index.

        Args:
            state: A numpy array containing the (x,y) coordinates of the state

        Returns:
            The integer index corresponding to the state coordinates
        """
        # Convert numpy array to tuple of integers
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        return state[0] * (self.state_ranges[0][1] - self.state_ranges[0][0]) + state[1]

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert a state index to its corresponding coordinate vector.

        Args:
            state_idx: The integer index of the state

        Returns:
            A numpy array containing the (x,y) coordinates corresponding to the state index
        """
        return np.array(
            [
                state_idx // (self.state_ranges[0][1] - self.state_ranges[0][0]),
                state_idx % (self.state_ranges[0][1] - self.state_ranges[0][0]),
            ]
        )

    def get_value_function(self) -> np.ndarray:
        """
        Get a deep copy of the current value function.

        Returns:
            A numpy array representing the value function table, where each entry
            maps a state index to its value
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v: np.ndarray) -> None:
        """
        Set the value function to a new array.

        Args:
            v: A numpy array containing the new value function table
        """
        self._value_function = copy.copy(v)

    def set_policy(self, state: int, action_prob_array: np.ndarray) -> None:
        """
        Set the action probability distribution for a given state.

        Args:
            state: The state index
            action_prob_array: A numpy array containing probabilities for each action
        """
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get the action probability distribution for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            A numpy array containing probabilities for each action in the given state
        """
        if isinstance(state, int):
            return self._policy[state]
        else:
            # Map state vector to state index
            return self._policy[self.get_state_index_from_coordinates(state)]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a deep copy of the current policy function.

        Returns:
            A numpy array representing the policy table, where each entry maps
            a state to a probability distribution over actions
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, p: np.ndarray) -> None:
        """
        Set the policy function to a new array.

        Args:
            p: A numpy array containing the new policy table
        """
        self._policy = copy.copy(p)


class GridworldSolver:
    """
    A solver for gridworld reinforcement learning problems.

    This class implements various policy computation methods for gridworld environments,
    including deterministic value iteration, stochastic policy iteration, and
    deterministic policy iteration.

    Attributes:
        _policy_type (str): Type of policy computation method to use
        env (gym.Env): The gridworld environment
        env_name (str): Name of the environment
        temperature (float): Temperature parameter for stochastic policies
        eps (float): Small constant for numerical stability
        gamma (float): Discount factor for future rewards
        solver (TabularPolicy): The policy object that stores computed policies and values
        performance_history (List[float]): History of cumulative rewards from policy evaluations
    """

    def __init__(
        self,
        policy_type: str = "deterministic_vi",
        gridworld_map_number: int = 0,
        noisy_transitions: bool = False,
    ) -> None:
        """
        Initialize the GridworldSolver.

        Args:
            policy_type: The type of policy computation to use. Must be one of:
                ["deterministic_vi", "stochastic_pi", "deterministic_pi"]
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions
            max_ent_temperature: Temperature parameter for stochastic policies

        Raises:
            AssertionError: If policy_type is not one of the allowed values
        """
        self._policy_type = policy_type
        assert policy_type in [
            "deterministic_vi",
            "stochastic_pi",
            "deterministic_pi",
            "exact_deterministic_pi",
        ]
        self.env: Optional[gym.Env] = None
        self.env_name = ""
        self.init_environment(gridworld_map_number, noisy_transitions)
        self.theta = 1e-3  # parameter to determine when to stop policy evaluation
        self.gamma = 0.99  # future return discount factor
        self.eps = 1e-8

        # Get the unwrapped environment to access its attributes
        assert self.env is not None
        unwrapped_env = self.env.unwrapped
        self.solver = TabularPolicy(
            unwrapped_env.num_states,
            unwrapped_env.get_state_ranges(),
            unwrapped_env.num_actions,
        )
        self.performance_history: List[float] = []

    def init_environment(
        self, gridworld_map_number: int = 0, noisy_transitions: bool = False
    ) -> None:
        """
        Initialize the gridworld environment.

        Args:
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions

        Raises:
            AssertionError: If gridworld_map_number is not 0 or 1
        """
        assert gridworld_map_number in [0, 1]
        if noisy_transitions:
            self.env_name = f"gridworldnoisy-v{gridworld_map_number}"
        else:
            self.env_name = f"gridworld-v{gridworld_map_number}"

        self.env = gym.make(self.env_name)
        self.env.reset()

    def compute_policy(self) -> None:
        """
        Compute optimal policy using the specified algorithm.

        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.
        """
        if self._policy_type == "deterministic_vi":
            self._value_iteration()
        elif self._policy_type == "stochastic_pi":
            self._stochastic_policy_iteration()
        elif self._policy_type == "exact_deterministic_pi":
            self._deterministic_policy_iteration_exact()
        else:  # deterministic_pi
            self._deterministic_policy_iteration()

    def solve(
        self,
        start_state: Optional[np.ndarray] = None,
        visualize: bool = False,
        max_steps: float = float("inf"),
    ) -> Tuple[float, int]:
        """
        Execute the current policy in the environment.

        This method runs the current policy from a given start state (or default start state)
        and returns the cumulative reward and number of steps taken.

        Args:
            start_state: Optional starting state coordinates
            visualize: Whether to render the environment
            max_steps: Maximum number of steps to take

        Returns:
            Tuple of (cumulative_reward, num_steps)
        """
        assert self.env is not None
        state, _ = self.env.reset()
        if start_state is not None:
            self.env.unwrapped.change_start_state(start_state)
            state = start_state

        if visualize:
            self.env.render()

        episode_reward = 0
        num_steps = 0
        done = False

        while not done and num_steps < max_steps:
            # Get action using current policy
            action = self.solver.get_action(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated

            episode_reward += reward
            num_steps += 1

            if visualize:
                self.env.render()

        return episode_reward, num_steps

    def plot_policy_curve(
        self, reward_history: List[float], filename: Optional[str] = None
    ) -> None:
        """
        Plot the learning curve showing policy performance over iterations.

        Args:
            reward_history: List of rewards from each policy evaluation
            filename: Optional path to save the plot
        """
        plt.figure()
        plt.plot(range(len(reward_history)), reward_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title("Policy Iteration Performance")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(figures_dir, "gridworld_learning_curve.png")

        plt.savefig(filename)
        plt.close()

    def plot_value_function(
        self, value_function: np.ndarray, filename: Optional[str] = None
    ) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap.

        Args:
            value_function: Array of state values to plot
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        # Normalize and reshape values
        V = (value_function - value_function.min()) / (
            value_function.max() - value_function.min() + self.eps
        )
        V = V.reshape(
            self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0],
            self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0],
        )

        V = np.flipud(V)

        # Create heatmap
        image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.0).astype(np.uint8)
        ax.set_title(f"Env: {self.env_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.imshow(image)

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._policy_type}_value.png"
            )
        plt.savefig(filename)
        plt.close()

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]

        return image, fig

    def plot_policy(
        self, policy: Optional[np.ndarray] = None, filename: Optional[str] = None
    ) -> None:
        """
        Plot the policy as arrows on the grid.

        Args:
            policy: Optional policy table to plot (defaults to solver policy)
            filename: Optional path to save the plot
        """
        if policy is None:
            policy = self.solver.get_policy_function()

        # Get dimensions of the grid
        rows = self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0]
        cols = self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0]

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(0, cols + 1, 1))
        ax.set_yticks(np.arange(0, rows + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        unwrapped_env = self.env.unwrapped

        # Arrow directions (assuming env.actions = [0:UP, 1:RIGHT, 2:DOWN, 3:LEFT])
        arrow_dict = {
            0: (0, 0),  # stay / no movement
            1: (0, 0.4),  # up
            2: (0, -0.4),  # down
            3: (-0.4, 0),  # left
            4: (0.4, 0),  # right
        }

        for s in range(self.solver.num_states):
            coord = self.solver.get_coordinates_from_state_index(s)
            row, col = coord
            # Flip vertically for plotting (so row 0 is at top)
            y = rows - row - 1
            x = col

            # Choose best action(s) for deterministic policy
            best_actions = np.where(policy[s] == np.max(policy[s]))[0]
            for a in best_actions:
                if a == 0:
                    continue
                dx, dy = arrow_dict[a]
                ax.arrow(
                    x + 0.5,
                    y + 0.5,
                    dx,
                    dy,
                    head_width=0.2,
                    head_length=0.2,
                    fc="k",
                    ec="k",
                )

        ax.set_title(f"Policy for {self.env_name}")
        plt.tight_layout()

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._policy_type}_policy.png"
            )

        plt.savefig(filename)
        plt.close()

    def _value_iteration(self) -> None:
        """
        Implement value iteration algorithm.

        This method iteratively updates state values based on the Bellman optimality
        equation until convergence. The student needs to implement:

        1. Value function update using the Bellman optimality equation:
           V(s) = max_a [ sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')] ]

        2. Policy update to be deterministic (probability 1 for best action):
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]

        3. Check for convergence by comparing old and new value functions

        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50  # run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        v_i = np.zeros(unwrapped_env.num_states)
        p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for k in range(horizon):
            print("Policy Iteration %d" % k)

            elapsed = time.time()
            # Student code here
            # Update value function
            v_old = v_i.copy()
            for s in range(unwrapped_env.num_states):
                q_vals = np.empty(unwrapped_env.num_actions)
                for a in unwrapped_env.actions:
                    q_vals[a] = np.sum(
                        T[s, a, s_next] * (R[s, a, s_next] + self.gamma * v_old[s_next]) for s_next in range(unwrapped_env.num_states)
                    )
                v_i[s] = np.max(q_vals)

            # Update policy
            # policy_stable = True
            for s in range(unwrapped_env.num_states):
                q_vals = np.empty(unwrapped_env.num_actions)
                for a in unwrapped_env.actions:
                    q_vals[a] = np.sum(
                        T[s, a, s_next] * (R[s, a, s_next] + self.gamma * v_i[s_next]) for s_next in range(unwrapped_env.num_states)
                    )
                best_a = int(np.argmax(q_vals))
                p_i[s] = np.eye(unwrapped_env.num_actions)[best_a]

            # Placeholder for student implementation
            # Check convergence
            elapsed = time.time() - elapsed
            print(".....Improve done in %g" % elapsed)
            self.performance_history.append(self.solve(max_steps=20)[0])
            if np.linalg.norm(v_i - v_old) < self.theta:
                break

        self.solver.set_policy_function(p_i)
        self.solver.set_value_function(v_i)

    def _deterministic_policy_iteration(self) -> None:
        """
        Implement deterministic policy iteration.

        This method alternates between policy evaluation and policy improvement steps,
        selecting the best action in each state deterministically. The student needs to implement:

        1. Policy Evaluation: Update value function using current policy:
           V(s) = sum_a pi(s,a) * sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           Note: Since policy is deterministic, this simplifies to:
           V(s) = sum_s' T(s,a*,s')[R(s,a*,s') + gamma * V(s')]
           where a* is the action with probability 1 in state s

        2. Policy Improvement: Update policy to be deterministic for best action:
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]

        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50  # run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        p_i = self.solver.get_policy_function()

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for k in range(50):
            print("Policy Iteration %d" % k)

            # Policy Evaluation
            elapsed = time.time()
            v_i = np.zeros(unwrapped_env.num_states)

            for _ in range(horizon):
                # Get expected value for current policy
                # Student code here
                for s in range(unwrapped_env.num_states):
                    a = np.argmax(p_i[s])
                    v_i[s] = sum(
                        T[s, a, s_next]
                        * (R[s, a, s_next] + self.gamma * v_i[s_next]) for s_next in range(unwrapped_env.num_states)
                    )

            elapsed = time.time() - elapsed
            print(".....Evaluate done in %g" % elapsed)
            elapsed = time.time()

            # Policy Improvement
            # Student code here
            for s in range(unwrapped_env.num_states):
                q_values = np.zeros(unwrapped_env.num_actions)
                for a in unwrapped_env.actions:
                    q_values[a] = sum(
                        T[s, a, s_next] * (R[s, a, s_next] + self.gamma * v_i[s_next]) for s_next in range(unwrapped_env.num_states)
                    )
                best_action = np.argmax(q_values)
                p_i[s] = np.eye(unwrapped_env.num_actions)[best_action]

            elapsed = time.time() - elapsed
            print(".....Improve done in %g" % elapsed)
            self.performance_history.append(self.solve(max_steps=20)[0])
            if np.array_equal(p_i, self.solver.get_policy_function()):
                break
            self.solver.set_policy_function(p_i)

        self.solver.set_value_function(v_i)

    def _eval_policy_exact(self, p_i: np.ndarray, T: np.ndarray, R: np.ndarray) -> np.ndarray:
        p_pi = np.zeros((self.env.unwrapped.num_states, self.env.unwrapped.num_states))
        r_pi = np.zeros(self.env.unwrapped.num_states)
        for s in range(self.env.unwrapped.num_states):
            a = int(np.argmax(p_i[s]))
            p_pi[s, :] = T[s, a, :]
            r_pi[s] = np.sum(T[s, a, :] * R[s, a, :])
        A = np.eye(self.env.unwrapped.num_states) - self.gamma * p_pi
        v = np.linalg.solve(A, r_pi)
        return v

    def _deterministic_policy_iteration_exact(self) -> None:
        T = np.zeros(
            (
                self.env.unwrapped.num_states,
                self.env.unwrapped.num_actions,
                self.env.unwrapped.num_states,
            )
        )
        R = np.zeros(
            (
                self.env.unwrapped.num_states,
                self.env.unwrapped.num_actions,
                self.env.unwrapped.num_states,
            )
        )
        p_i = self.solver.get_policy_function()
        for s in range(self.env.unwrapped.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in self.env.unwrapped.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    self.env.unwrapped.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = self.env.unwrapped.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = self.env.unwrapped.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )
        for k in range(50):
            start = time.time()
            v_i = self._eval_policy_exact(p_i, T, R)
            print(".....Exact Evaluate done in %g" % (time.time() - start))

            start = time.time()
            new_p = p_i.copy()
            for s in range(self.env.unwrapped.num_states):
                q = np.zeros(self.env.unwrapped.num_actions)
                for a in self.env.unwrapped.actions:
                    q[a] = np.sum(T[s, a, :] * (R[s, a, :] + self.gamma * v_i))
                best = int(np.argmax(q))
                new_p[s] = np.eye(self.env.unwrapped.num_actions)[best]
            print(".....Improve done in %g" % (time.time() - start))

            self.performance_history.append(self.solve(max_steps=20)[0])
            if np.array_equal(new_p, p_i):
                break
            p_i = new_p

        self.solver.set_policy_function(p_i)
        self.solver.set_value_function(v_i)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    p_i_times = []
    v_i_times = []

    ############ Q1.1 ############
    gw0_solver = GridworldSolver(policy_type="deterministic_pi", gridworld_map_number=0)
    gw1_solver = GridworldSolver(policy_type="deterministic_pi", gridworld_map_number=1)
    for i, solver in enumerate([gw0_solver, gw1_solver]):
        gamma_vals = [0.5, 0.75, 0.9, 0.99]
        for val in gamma_vals:
            solver.gamma = val
            print(f"Starting for gamma={val}!")
            policy_filename = os.path.join(
                figures_dir,
                f"{solver.env_name}_{solver._policy_type}_gamma={solver.gamma}_policy.png",
            )
            value_filename = os.path.join(
                figures_dir,
                f"{solver.env_name}_{solver._policy_type}_gamma={solver.gamma}_value.png",
            )
            curve_filename = os.path.join(
                figures_dir,
                f"{solver.env_name}_{solver._policy_type}_gamma={solver.gamma}_curve.png",
            )
            start_time = time.time()
            solver.compute_policy()
            elapsed_time = time.time() - start_time
            p_i_times.append(elapsed_time)
            print("Computed Q2 PI Policy in %g seconds" % elapsed_time)
            solver.plot_policy_curve(
                solver.performance_history, curve_filename
            )
            solver.plot_value_function(solver.solver.get_value_function(), value_filename)
            solver.plot_policy(filename=policy_filename)
    gw0_exact_solver = GridworldSolver(
        policy_type="exact_deterministic_pi", gridworld_map_number=0
    )
    gw1_exact_solver = GridworldSolver(
        policy_type="exact_deterministic_pi", gridworld_map_number=1
    )
    for i, solver in enumerate([gw0_exact_solver, gw1_exact_solver]):
        policy_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_policy.png",
        )
        value_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_value.png",
        )
        curve_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_curve.png",
        )
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        p_i_times.append(elapsed_time)
        print("Computed Q2 PI Policy Exact in %g seconds" % elapsed_time)
        solver.plot_policy_curve(
            solver.performance_history, curve_filename
        )
        solver.plot_value_function(solver.solver.get_value_function(), value_filename)
        solver.plot_policy(filename=policy_filename)

    ########### Q1.2 ############
    gw0_det_solver = GridworldSolver(
        policy_type="deterministic_vi", gridworld_map_number=0
    )
    gw1_det_solver = GridworldSolver(
        policy_type="deterministic_vi", gridworld_map_number=1
    )
    for solver in [gw0_det_solver, gw1_det_solver]:
        policy_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_policy.png",
        )
        value_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_value.png",
        )
        curve_filename = os.path.join(
            figures_dir,
            f"{solver.env_name}_{solver._policy_type}_curve.png",
        )
        print("Starting!")
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        v_i_times.append(elapsed_time)
        print("Computed Q1.a VI Policy in %g seconds" % elapsed_time)
        solver.plot_value_function(solver.solver.get_value_function(), value_filename)
        solver.plot_policy(filename=policy_filename)
        solver.plot_policy_curve(solver.performance_history, curve_filename)

    def _compare_times(n):
        """
        This function runs a basic monte carlo simulation to compare the computation times
        of value iteration vs policy iteration vs exact solutions.
        """
        times={"deterministic_vi": [], "deterministic_pi": [], "exact_deterministic_pi": []}
        for _ in range(n):
            for solver in [gw0_solver, gw1_solver, gw0_det_solver, gw1_det_solver, gw0_exact_solver, gw1_exact_solver]:
                start_time = time.time()
                solver.compute_policy()
                elapsed_time = time.time() - start_time
                times[solver._policy_type].append(elapsed_time)

        vi = np.asarray(times["deterministic_vi"], dtype=float)
        pi = np.asarray(times["deterministic_pi"], dtype=float)
        exact_pi = np.asarray(times["exact_deterministic_pi"], dtype=float)
        v_mean, v_std = vi.mean(), vi.std()
        v_mask = np.ones_like(vi, dtype=bool) if v_std == 0 else (np.abs(vi - v_mean) <= 3 * v_std)
        filtered_vi = vi[v_mask]
        p_mean, p_std = pi.mean(), pi.std()
        p_mask = np.ones_like(pi, dtype=bool) if p_std == 0 else (np.abs(pi - p_mean) <= 3 * p_std)
        filtered_pi = pi[p_mask]
        p_exact_mean, p_exact_std = exact_pi.mean(), exact_pi.std()
        p_exact_mask = (
            np.ones_like(exact_pi, dtype=bool)
            if p_exact_std == 0
            else (np.abs(exact_pi - p_exact_mean) <= 3 * p_exact_std)
        )
        filtered_exact_pi = exact_pi[p_exact_mask]
        with open(os.path.join(figures_dir, "computation_times.tex"), "w") as f:
            f.write(f"\\newcommand{{\\ViSTD}}{{{v_std:.4f}}}\n")
            f.write(f"\\newcommand{{\\ViMean}}{{{v_mean:.4f}}}\n")
            f.write(f"\\newcommand{{\\PiSTD}}{{{p_std:.4f}}}\n")
            f.write(f"\\newcommand{{\\PiMean}}{{{p_mean:.4f}}}\n")
            f.write(f"\\newcommand{{\\PiExactSTD}}{{{p_exact_std:.4f}}}\n")
            f.write(f"\\newcommand{{\\PiExactMean}}{{{p_exact_mean:.4f}}}\n")
        plt.hist(filtered_vi)
        plt.xlabel("Computation Time (s)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(figures_dir, "vi_time_histogram.png"))
        plt.close()
        plt.hist(filtered_pi)
        plt.xlabel("Computation Time (s)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(figures_dir, "pi_time_histogram.png"))
        plt.close()
        plt.hist(filtered_exact_pi)
        plt.xlabel("Computation Time (s)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(figures_dir, "exact_pi_time_histogram.png"))
        plt.close()
    _compare_times(100)
