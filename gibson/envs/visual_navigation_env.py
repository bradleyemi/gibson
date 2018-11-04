from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.envs.cube_projection import Cube, generate_projection_matrix, draw_cube
import numpy as np
from scipy.misc import imresize
import pickle
import math
from copy import copy

class HuskyVisualNavigateEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None):
        HuskyNavigateEnv.__init__(self, config, gpu_count)
        self.min_target_x, self.max_target_x = self.config["x_target_range"]
        self.min_target_y, self.max_target_y = self.config["y_target_range"]
        self.min_agent_x, self.max_agent_x = self.config["x_boundary"]
        self.min_agent_y, self.max_agent_y = self.config["y_boundary"]
        self.min_spawn_x, self.max_spawn_x = self.config["x_spawn_range"]
        self.min_spawn_y, self.max_spawn_y = self.config["y_spawn_range"]
        self.default_z = self.config["initial_pos"][2]
        self.cube_size = 0.2
        self.target_radius = 0.5
        self.cube_image = np.zeros((self.config["resolution"], self.config["resolution"], 3), np.uint32)
        self.target_x = np.random.uniform(self.min_target_x, self.max_target_x)
        self.target_y = np.random.uniform(self.min_target_y, self.max_target_y)
        self.min_target_distance = self.config["min_target_distance"]
        self.max_target_distance = self.config["max_target_distance"]

    def _close_to_goal(self):
        target_vector = np.array([self.target_x, self.target_y])
        return np.linalg.norm(self.robot.get_position()[0:2] - target_vector) < self.target_radius

    def _rewards(self, action=None, debugmode=False):
        x, y, z = self.robot.get_position()
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        dead_penalty = 0.0
        if dead:
            dead_penalty = -10.0
        
        close_to_goal = 0.
        if self._close_to_goal():
            close_to_goal = 10.
        alive = -0.025
        return [alive, dead_penalty, close_to_goal]
    
    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        obs["rgb_filled"] = self._add_cube_into_image(obs)
        return obs, rew, done, info

    def _add_cube_into_image(self, obs):
        cube_x = self.target_x
        cube_y = self.target_y
        cube_z = self.default_z
        cube = Cube(origin=np.array([cube_x, cube_y, cube_z]), scale=self.cube_size)
        roll, pitch, yaw = self.robot.get_rpy()
        x, y, z = self.robot.eyes.get_position()
        size = self.config["resolution"] // 2
        fov = self.config["fov"]
        world_to_image_mat = generate_projection_matrix(x, y, z, yaw, pitch, roll, fov, fov, size, size)
        cube_idx = draw_cube(cube, world_to_image_mat, size*2, size*2, fast_depth=True)
        depth = copy(obs["depth"])
        depth[depth == 0] = np.inf
        self.cube_image = copy(obs["rgb_filled"])
        self.cube_image[cube_idx < depth.squeeze()] = np.array([0, 255, 0])
        return self.cube_image

    def _reset(self):
        obs = HuskyNavigateEnv._reset(self)
        obs["rgb_filled"] = self._add_cube_into_image(obs)
        
        self.target_x = np.random.uniform(self.min_target_x, self.max_target_x)
        self.target_y = np.random.uniform(self.min_target_y, self.max_target_y)
        while True:
            spawn_x = np.random.uniform(self.min_spawn_x, self.max_spawn_x)
            spawn_y = np.random.uniform(self.min_spawn_y, self.max_spawn_y)
            distance = np.linalg.norm(np.array([spawn_x, spawn_y]) - np.array([self.target_x, self.target_y]))
            if distance < self.max_target_distance and distance > self.min_target_distance:
                break
        self.config["initial_pos"] = [spawn_x, spawn_y, self.default_z]
        return obs

    def _termination(self, debugmode=False):
        # some checks to make sure the husky hasn't flipped or fallen off
        x, y, z = self.robot.get_position()
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        if (abs(roll) > 1.22):
            print("Agent roll too high")
        if (abs(pitch) > 1.22):
            print("Agent pitch too high")
        if (abs(z - z_initial) > 0.5):
            print("Agent fell off")
        out_of_bounds = x < self.min_agent_x or x > self.max_agent_x or y < self.min_agent_y or y > self.max_agent_y
        dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        done = dead or out_of_bounds or self.nframe >= self.config["episode_length"] or self._close_to_goal()
        return done


    def render_map(self):
        return self.cube_image

    def render_map_rgb(self):
        x = self.render_map()
        return x