from gibson.envs.visual_navigation_env import HuskyVisualNavigateEnv, HuskyCoordinateNavigateEnv, HuskyCoordinateNavigateMultiEnv
from gibson.envs.exploration_env import HuskyVisualExplorationEnv
from gibson.utils.play import play
import os
import yaml
from gibson.data.datasets import get_model_path

DEFAULT_CONFIGS = {
    'navigation': '../configs/husky_visual_navigate.yaml',
    'exploration': '../configs/husky_visual_explore.yaml',
    'local_planning': '../configs/husky_coordinate_navigate.yaml',
    'local_planning_multi': '/home/bradleyemi/svl/visual-cortex-parent/teas/teas/env/gibson/husky_coordinate_navigate_multienv.yaml'
}

def make_env(downstream_task, config=None):
    assert (downstream_task in ['navigation', 'exploration', 'local_planning', 'local_planning_multi'])
    if config is None:
        config = DEFAULT_CONFIGS[downstream_task]
    start_locations_file = os.path.join(get_model_path('Beechwood'), "first_floor_poses.csv")
    if downstream_task == 'navigation':
        return HuskyVisualNavigateEnv(config=config, gpu_count=0)
    elif downstream_task == 'exploration':
        return HuskyVisualExplorationEnv(config=config, gpu_count=0, start_locations_file=start_locations_file)
    elif downstream_task == 'local_planning':
        return HuskyCoordinateNavigateEnv(config=config, gpu_count=0, start_locations_file=start_locations_file)
    elif downstream_task == 'local_planning_multi':
        return HuskyCoordinateNavigateMultiEnv(config=config, gpu_count=0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--downstream_task', type=str) # choose "navigation", "exploration", "local_planning"
    args = parser.parse_args()
    env = make_env(args.downstream_task, config=args.config)
    play(env)