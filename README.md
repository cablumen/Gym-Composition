# Gym-Composition
This repo is an exploration of idea of composition with neural networks trained with gym environments.

Environment info (from https://github.com/openai/gym/blob/6e2e921b5faaa9c356f589e4e938e718824c7d4a/gym/envs/classic_control/cartpole.py)
    Action Space
        The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
        of the fixed force the cart is pushed with.

        | Num | Action                 |
        |-----|------------------------|
        | 0   | Push cart to the left  |
        | 1   | Push cart to the right |
    
    Observation Space
        The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | Cart Position         | -4.8                | 4.8               |
        | 1   | Cart Velocity         | -Inf                | Inf               |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
        | 3   | Pole Angular Velocity | -Inf                | Inf               |

    Rewards
        Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is 475 for v1.

TODO:
    Test reward arch impact on reward
    Move AtomicAction train logging to evulate based on step
    Change get/set logic so it's directly invoked instead of through helper methods
