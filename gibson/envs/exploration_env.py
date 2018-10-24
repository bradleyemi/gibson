from gibson.envs.husky_env import HuskyNavigateEnv
import numpy as np
from scipy.misc import imresize
import pickle

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
            return (0,-1)

    def _rewards(self, action=None, debugmode=False):
        position = self.robot.get_position()
        x, y = position[0:2]
        orientation = self.robot.get_rpy()[2]
        quadrant = self.get_quadrant(orientation)
        x_idx = np.searchsorted(self.x_vals, x)
        y_idx = np.searchsorted(self.y_vals, y)
        new_block = 0.
        if (x_idx + quadrant[0] >= self.x_vals.shape[0]) or (x_idx + quadrant[0] < 0) or \
           (y_idx + quadrant[1] >= self.y_vals.shape[0]) or (y_idx + quadrant[1] < 0):
           return [0.]
        if self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] == 0:
            self.occupancy_map[x_idx + quadrant[0], y_idx + quadrant[1]] = 1.
            new_block = 1.
        return [new_block]
    
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

