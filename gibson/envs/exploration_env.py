from gibson.envs.husky_env import HuskyNavigateEnv
import numpy as np
from scipy.misc import imresize
import pickle
import math


class HuskyExplorationEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None):
        HuskyNavigateEnv.__init__(self, config, gpu_count)
        self.cell_size = self.config["cell_size"]
        self.map_x_range = self.config["map_x_range"]
        self.map_y_range = self.config["map_y_range"]

        self.x_vals = np.arange(self.map_x_range[0], self.map_x_range[1], self.cell_size)
        self.y_vals = np.arange(self.map_y_range[0], self.map_y_range[1], self.cell_size)
        self.occupancy_map = np.zeros((self.x_vals.shape[0], self.y_vals.shape[0]))
        
        self.start_locations_file = start_locations_file
        if self.start_locations_file is not None:
            with open(self.start_locations_file, 'r') as f:
                self.points = np.loadtxt(f, delimiter=',')
                self.n_points = self.points.shape[0]

    def get_quadrant(self, angle):
        if angle > np.pi / 4 and angle <= 3 * np.pi / 4:
            return (0,1)
        elif angle > -np.pi / 4 and angle <= np.pi / 4:
            return (1,0)
        elif angle > -3 * np.pi / 4 and angle <= -np.pi / 4:
            return (0,-1)
        else:
            return (-1,0)

    def _rewards(self, action=None, debugmode=False):
        position = self.robot.get_position()
        x, y = position[0:2]
        orientation = self.robot.get_rpy()[2]
        quadrant = self.get_quadrant(orientation)
        x_idx = int((x - self.map_x_range[0]) / self.cell_size)
        y_idx = int((y - self.map_y_range[0]) / self.cell_size)
        new_block = 0.
        if (x_idx + quadrant[0] >= self.x_vals.shape[0]) or (x_idx + quadrant[0] < 0) or \
           (y_idx + quadrant[1] >= self.y_vals.shape[0]) or (y_idx + quadrant[1] < 0):
           return [0.]
        if self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] == 0:
            self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] = 1.
            new_block = 1.

        z = self.robot.get_position()[2]
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        death = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        if death:
            death_penalty = -10.0
        else:
            death_penalty = 0.0

        return [new_block, death_penalty]
    
    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        obs["map"] = self.render_map()
        return obs, rew, done, info

    def _reset(self):
        if self.start_locations_file is not None:
            new_start_location = self.points[np.random.randint(self.n_points), :]
            self.config["initial_pos"] = [new_start_location[0], new_start_location[1], new_start_location[2]]
        obs = HuskyNavigateEnv._reset(self)
        self.occupancy_map = np.zeros((self.x_vals.shape[0], self.y_vals.shape[0]))
        obs["map"] = self.render_map()
        return obs

    def _termination(self, debugmode=False):
        # some checks to make sure the husky hasn't flipped or fallen off
        z = self.robot.get_position()[2]
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        if (abs(roll) > 1.22):
            print("Agent roll too high")
        if (abs(pitch) > 1.22):
            print("Agent pitch too high")
        if (abs(z - z_initial) > 0.5):
            print("Agent fell off")
        done = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5 or self.nframe >= self.config["episode_length"]
        return done


    def render_map(self):
        x = self.occupancy_map * 255.
        x = x.astype(np.uint8)
        x = imresize(x, (self.config["resolution"], self.config["resolution"]))
        return x

    def render_map_rgb(self):
        x = self.render_map()
        return np.repeat(x[:,:,np.newaxis], 3, axis=2)


class HuskyVisualExplorationEnv(HuskyExplorationEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None):
        HuskyExplorationEnv.__init__(self, config, gpu_count, start_locations_file)
        self.min_depth = 0.
        self.max_depth = 2.5
        self.fov = self.config["fov"]
        self.screen_dim = self.config["resolution"]

    def _step(self, action):
        orig_found = np.sum(self.occupancy_map)
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        self._update_occupancy_map(obs['depth'])
        rew = np.sum(self.occupancy_map) - orig_found
        obs["map"] = self.render_map()
        self.reward += rew
        self.eps_reward += rew
        return obs, rew, done, info

    def _update_occupancy_map(self, depth_image):
        clipped_depth_image = np.clip(depth_image, self.min_depth, self.max_depth)
        xyz = self._reproject_depth_image(depth_image.squeeze())
        xx, yy = self.rotate_origin_only(xyz[self.screen_dim//2:, self.screen_dim//2, :], math.radians(90) - self.robot.get_rpy()[2])
        xx += self.robot.get_position()[0]
        yy += self.robot.get_position()[1]
        for x, y in zip(xx, yy):
            self.insert_occupancy_map(x, y)

    def insert_occupancy_map(self, x, y):
        idx_x = int((x - self.map_x_range[0]) / self.cell_size)
        idx_y = int((y - self.map_y_range[0]) / self.cell_size)
        idx_x = np.clip(idx_x, 0, self.x_vals.shape[0] - 1)
        idx_y = np.clip(idx_y, 0, self.y_vals.shape[0] - 1)
        if idx_y < 0 or idx_y < 0:
            raise ValueError("Trying to set occupancy in grid cell ({}, {})".format(idx_x, idx_y))
        self.occupancy_map[idx_x, idx_y] = 1
        return idx_x, idx_y


    def _reproject_depth_image(self, depth, unit_scale=1.0):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.
        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        y = depth * unit_scale
        x = y * ((c - self.screen_dim // 2) / self.fov / self.screen_dim // 2)
        z = y * ((r - self.screen_dim // 2) / self.fov / self.screen_dim // 2)
        return np.dstack((x, y, z))

    def rotate_origin_only(self, xy, radians):
        x, y = xy[:,:2].T
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)
        return xx, yy



