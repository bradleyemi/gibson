from gibson.envs.exploration_env import HuskyVisualExplorationEnv
from gibson.utils.play import play
import os

config_file = "/home/bradleyemi/svl/visual-cortex-parent/teas/teas/env/gibson/husky_visual_explore_train.yaml"
#start_locations_file = "/home/bradleyemi/svl/visual-cortex-parent/teas/teas/env/gibson/Wiconisco_start_locations.csv"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyVisualExplorationEnv(config=args.config, gpu_count = 0, start_locations_file=None)
    play(env, zoom=4)