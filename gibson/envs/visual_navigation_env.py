from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.envs.cube_projection import Cube, generate_projection_matrix, draw_cube
import numpy as np
from scipy.misc import imresize
import pickle
import math
from copy import copy


class HuskyVisualNavigateEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, valid_locations=None):
        HuskyNavigateEnv.__init__(self, config, gpu_count)
        self.use_valid_locations = valid_locations is not None
        if self.use_valid_locations:
            print("Using valid locations!")
            self.valid_locations = self.get_valid_locations(valid_locations)
            self.n_locations = self.valid_locations.shape[0]
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
    
    def get_valid_locations(self, valid_locations):
        return np.loadtxt(valid_locations, delimiter=',')

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
        alive = 0.0
        return [alive, dead_penalty, close_to_goal]
    
    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        self.depth = copy(obs["depth"]).squeeze()
        self.depth[self.depth == 0] = np.inf
        obs["rgb_filled"] = self._add_cube_into_image(obs, [self.target_x, self.target_y, self.default_z])
        return obs, rew, done, info

    def _add_cube_into_image(self, obs, location, color=[20,200,200]):
        cube = Cube(origin=np.array(location), scale=self.cube_size)
        self.cube_image = copy(obs["rgb_filled"])
        roll, pitch, yaw = self.robot.get_rpy()
        x, y, z = self.robot.eyes.get_position()
        size = self.config["resolution"] // 2
        fov = self.config["fov"]
        world_to_image_mat = generate_projection_matrix(x, y, z, yaw, pitch, roll, fov, fov, size, size)
        cube_idx = draw_cube(cube, world_to_image_mat, size*2, size*2, fast_depth=True)
        self.cube_image[cube_idx < self.depth] = np.array(color)
        return self.cube_image
    
    def sample_valid_location(self):
        index = np.random.choice(range(self.n_locations))
        return self.valid_locations[index,:]

    def select_new_target(self):
        if self.use_valid_locations:
            location = self.sample_valid_location()
            print("New target is:", location)
            return location
        else:
            return np.random.uniform(self.min_target_x, self.max_target_x), np.random.uniform(self.min_target_y, self.max_target_y)

    def select_new_agent_location(self):
        attempts = 0
        while attempts < 100:
            if self.use_valid_locations:
                spawn_x, spawn_y = self.sample_valid_location()
            else:
                spawn_x = np.random.uniform(self.min_spawn_x, self.max_spawn_x)
                spawn_y = np.random.uniform(self.min_spawn_y, self.max_spawn_y)
            distance = np.linalg.norm(np.array([spawn_x, spawn_y]) - np.array([self.target_x, self.target_y]))
            if distance < self.max_target_distance and distance > self.min_target_distance:
                break
            attempts += 1
        if attempts == 100:
            raise Exception("Could not find a valid spawn location. Try expanding the target distance range.")
        print("New agent location is:", spawn_x, spawn_y)
        return [spawn_x, spawn_y, self.default_z]

    def _reset(self):
        obs = HuskyNavigateEnv._reset(self)
        self.cube_image = copy(obs["rgb_filled"])
        self.depth = copy(obs["depth"]).squeeze()
        self.target_x, self.target_y = self.select_new_target()
        obs["rgb_filled"] = self._add_cube_into_image(obs, [self.target_x, self.target_y, self.default_z])
        self.config["initial_pos"] = self.select_new_agent_location()
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
        #out_of_bounds = x < self.min_agent_x or x > self.max_agent_x or y < self.min_agent_y or y > self.max_agent_y
        dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        done = dead or self.nframe >= self.config["episode_length"] or self._close_to_goal()
        return done


    def render_map(self):
        return self.cube_image

    def render_map_rgb(self):
        x = self.render_map()
        return x

class HuskyVisualObstacleAvoidanceEnv(HuskyVisualNavigateEnv):
    def __init__(self, config, gpu_count=0, start_locations_file=None):
        HuskyVisualNavigateEnv.__init__(self, config, gpu_count)
        self.num_obstacles = self.config["num_obstacles"]
        self.min_obstacle_separation = self.config["min_obstacle_separation"]
        self.target_color = [0, 159, 107]
        self.obstacle_color = [196, 2, 51]
        self.obstacle_radius = 0.5
        # spawn obstacles:
        self.reset_cube_locations()

    def _add_target_into_image(self, obs, location, color=[0, 159, 107]):
        cube = Cube(origin=np.array(location), scale=self.cube_size)
        roll, pitch, yaw = self.robot.get_rpy()
        x, y, z = self.robot.eyes.get_position()
        size = self.config["resolution"] // 2
        fov = self.config["fov"]
        world_to_image_mat = generate_projection_matrix(x, y, z, yaw, pitch, roll, fov, fov, size, size)
        self.target_depth = draw_cube(cube, world_to_image_mat, size*2, size*2, fast_depth=True)
        self.cube_image[self.target_depth < self.depth] = np.array(color)
        return self.cube_image

    def _add_obstacle_into_image(self, obs, location, color=[0, 159, 107]):
        cube = Cube(origin=np.array(location), scale=self.cube_size)
        roll, pitch, yaw = self.robot.get_rpy()
        x, y, z = self.robot.eyes.get_position()
        size = self.config["resolution"] // 2
        fov = self.config["fov"]
        world_to_image_mat = generate_projection_matrix(x, y, z, yaw, pitch, roll, fov, fov, size, size)
        cube_idx = draw_cube(cube, world_to_image_mat, size*2, size*2, fast_depth=True)
        self.cube_image[np.logical_and(cube_idx < self.depth, cube_idx < self.target_depth)] = np.array(color)
        return self.cube_image

    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        self.cube_image = copy(obs["rgb_filled"])
        self.depth = copy(obs["depth"]).squeeze()
        self.depth[self.depth == 0] = np.inf
        obs["rgb_filled"] = self._add_target_into_image(obs, [self.target_x, self.target_y, self.default_z], color=self.target_color)
        for location in self.cube_locations:
            obs["rgb_filled"] = self._add_obstacle_into_image(obs, location, color=self.obstacle_color)
        return obs, rew, done, info

    def _rewards(self, action=None, debugmode=False):
        x, y, z = self.robot.get_position()
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        dead_penalty = 0.0
        if dead:
            dead_penalty = 0.0
        
        close_to_goal = 0.
        if self._close_to_goal():
            close_to_goal = 10.
        alive = -0.025

        close_to_obstacles = 0.
        if self.close_to_obstacles():
            close_to_obstacles = -10.0

        return [alive, dead_penalty, close_to_obstacles, close_to_goal]
    
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
        close_to_obstacles = self.close_to_obstacles()
        done = dead or out_of_bounds or close_to_obstacles or self.nframe >= self.config["episode_length"] or self._close_to_goal()
        return done

    def _reset(self):
        obs = HuskyNavigateEnv._reset(self)
        self.cube_image = copy(obs["rgb_filled"])
        self.depth = copy(obs["depth"]).squeeze()
        attempts = 0
        #print("Generating target and spawn locations")
        while attempts < 100:
            self.target_x = np.random.uniform(self.min_target_x, self.max_target_x)
            self.target_y = np.random.uniform(self.min_target_y, self.max_target_y)
            spawn_x = np.random.uniform(self.min_spawn_x, self.max_spawn_x)
            spawn_y = np.random.uniform(self.min_spawn_y, self.max_spawn_y)
            distance = np.linalg.norm(np.array([spawn_x, spawn_y]) - np.array([self.target_x, self.target_y]))
            if distance < self.max_target_distance and distance > self.min_target_distance:
                break
            attempts += 1
        if attempts == 100:
            raise Exception("Agent could not be spawned. Try expanding the min and max target range.")
        self.config["initial_pos"] = [spawn_x, spawn_y, self.default_z]
        obs["rgb_filled"] = self._add_target_into_image(obs, [self.target_x, self.target_y, self.default_z], color=self.target_color)
        #print("Generating obstacles")
        self.reset_cube_locations()
        x, y = self.config["initial_pos"][0:2]
        for location in self.cube_locations:
            obs["rgb_filled"] = self._add_obstacle_into_image(obs, location, color=self.obstacle_color)
        return obs

    def close_to_obstacles(self):
        x, y, z = self.robot.get_position()
        for location in self.cube_locations:
            if np.linalg.norm(np.array([x,y]) - np.array(location[0:2])) < self.obstacle_radius:
                return True
        return False

    def reset_cube_locations(self):
        attempts = 0
        while attempts < 10:
            #print("Attempt number", attempts, "at resetting cube locations")
            attempts += 1
            try:
                self.cube_locations = []
                for i in range(self.num_obstacles):
                    #print(self.cube_locations)
                    new_location = self.get_new_cube_spawn_location(self.cube_locations)
                    self.cube_locations.append(new_location)   
                break
            except:
                continue 
            
        if attempts == 10:
            raise Exception("Could not reset the cube locations.")  

    def get_new_cube_spawn_location(self, cubes):
        attempts = 0
        while attempts < 100:
            #print("attempt number", attempts, "at spawning this cube")
            attempts += 1
            spawn_cube_x = np.random.uniform(self.min_spawn_x, self.max_spawn_x)
            spawn_cube_y = np.random.uniform(self.min_spawn_y, self.max_spawn_y)
            # Make sure it's within correct range of agent
            dist_to_agent = np.linalg.norm(np.array(self.config["initial_pos"][0:2]) - np.array([spawn_cube_x, spawn_cube_y]))
            if dist_to_agent < self.min_obstacle_separation:
                continue
            # Make sure it's not too close to target
            dist_to_target = np.linalg.norm(np.array([self.target_x, self.target_y]) - np.array([spawn_cube_x, spawn_cube_y]))
            if dist_to_target < self.min_obstacle_separation:
                continue
            # Make sure it's not too close to other cubes
            dist_to_cubes = [np.linalg.norm(np.array([cube[0], cube[1]]) - np.array([spawn_cube_x, spawn_cube_y])) for cube in cubes]
            if (len(dist_to_cubes) > 0) and min(dist_to_cubes) < self.min_obstacle_separation:
                continue
            break
        
        if attempts == 100:
            raise Exception("Too many attempts at spawning cubes, increase the spawn area or decrease num obstacles.")

        return [spawn_cube_x, spawn_cube_y, self.default_z]   
