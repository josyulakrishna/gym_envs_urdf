from gym.envs.registration import register
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.urdf_env2 import UrdfEnv2
register(
    id='urdf-env-v0',
    entry_point='urdfenvs.urdf_common:UrdfEnv'
)
register(
    id='urdf-env-v1',
    entry_point='urdfenvs.urdf_common:UrdfEnv2'
)
