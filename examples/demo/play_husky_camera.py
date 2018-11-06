from gibson.envs.visual_navigation_env import HuskyVisualNavigateEnv
from gibson.utils.play import play
import os

config_file = "/home/bradleyemi/svl/visual-cortex-parent/teas/teas/env/gibson/husky_visual_navigate.yaml"
start_locations_file = "/home/bradleyemi/svl/visual-cortex-parent/GibsonEnv/gibson/assets/dataset/Beechwood/first_floor_poses.csv"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyVisualNavigateEnv(config=args.config, gpu_count = 0, valid_locations=start_locations_file)
    play(env, zoom=4)