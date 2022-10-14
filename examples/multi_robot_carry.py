import gym
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle


def create_obstacle():
    """
    You can place any arbitrary urdf file in here.
    The physics of this box is not perfect, but this is not the field of
    my expertise.
    """
    urdf_obstacle_dict = {
        "type": "urdf",
        "geometry": {"position": [0.2, -0.0, 1.05]},
        "urdf": os.path.join(os.path.dirname(__file__), "block.urdf"),
    }
    urdf_obstacle = UrdfObstacle(name="carry_object", content_dict=urdf_obstacle_dict)
    return urdf_obstacle


def run_multi_robot_carry(n_steps=1000, render=False):
    # Two robots to carry the load
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=render)
    # Choosing arbitrary actions
    action = [0.2, 0.0, 0.0, 0.2, -0.11, 0.0]
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    ob = env.reset(base_pos=base_pos)
    # Placing the obstacle
    env.add_obstacle(create_obstacle())

    history = []
    for _ in range(n_steps):
        # WARNING: The reward function is not defined for you case.
        # You will have to do this yourself.
        ob, reward, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    run_multi_robot_carry(render=True)
