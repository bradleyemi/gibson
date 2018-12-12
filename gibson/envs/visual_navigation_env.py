from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.envs.cube_projection import Cube, generate_projection_matrix, draw_cube, get_cube_depth_and_faces
import numpy as np
from scipy.misc import imresize
from skimage.io import imread
import skimage
import pickle
import math
from copy import copy
import os
import matplotlib.pyplot as plt
from gibson import assets
import meshcut

class OccupancyMap(object):
    
    def __init__(self, xmin, xmax, ymin, ymax, voxel_length):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.voxel_length = voxel_length      
        self.size_x = math.ceil((xmax - xmin) / voxel_length)
        self.size_y = math.ceil((ymax - ymin) / voxel_length)
        self.bitmap = np.full((self.size_x, self.size_y), False, dtype=np.bool_)
        
    def update(self, x, y, val=True, thickness=0):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        self.bitmap[idx_x, idx_y] = val
        for t in range(thickness):
            for xc in range(idx_x - t, idx_x + t):
                for yc in range(idx_y - t, idx_y + t):
                    self.bitmap[xc, yc] = val 
        return self.bitmap
    
    def get(self, x, y):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        return self.bitmap[idx_x, idx_y]

    def clear(self):
        self.bitmap = np.full((self.size_x, self.size_y), False, dtype=np.bool_)

    def _get_voxel_coords(self, x, y):
        idx_x = int((x - self.xmin) / self.voxel_length)
        idx_y = int((y - self.ymin) / self.voxel_length)
        idx_x = np.clip(idx_x, 0, self.size_x - 1)
        idx_y = np.clip(idx_y, 0, self.size_y - 1)
        if idx_y < 0 or idx_y < 0:
            raise ValueError("Trying to set occupancy in grid cell ({}, {})".format(idx_x, idx_y))
        return idx_x, idx_y

def load_obj(fn):
    verts = []
    faces = []
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

class NavigationMapRenderer(object):
    def __init__(self, x_min, x_max, y_min, y_max, mesh_file, mesh_z, voxel_size, line_resolution=50, render_resolution=128):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.mesh_file = mesh_file
        self.mesh_z = mesh_z
        self.voxel_size = voxel_size
        self.line_resolution = line_resolution
        self.render_resolution = render_resolution
        self.map_color = np.array([0,0,0])
        self.agent_color = np.array([0,0,255])
        self.target_color = np.array([0,255,0])
        self.spawn_color = np.array([255,0,0])
        self.last_pose = None
        self.create_map_layer(self.mesh_file, self.mesh_z)
        self.create_target_layer()
        self.create_spawn_layer()
        self.create_agent_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def create_map_layer(self, mesh_file, mesh_z):
        print("Creating map layer...")
        self.map_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)
        verts, faces = load_obj(mesh_file)
        cross_section = meshcut.cross_section(verts, faces, plane_orig=(0,0,mesh_z), plane_normal=(0,0,1))
        for item in cross_section:
            xy = item[:,0:2]
            for i in range(xy.shape[0] - 1):
                xl = xy[i,0]
                xh = xy[i+1,0]
                yl = xy[i,1]
                yh = xy[i+1,1]
                self.draw_line(self.map_layer, xl, yl, xh, yh, self.line_resolution)
        print("Done.")
        
    def draw_map(self):
        self.render_image[self.map_layer.bitmap] = self.map_color

    def create_target_layer(self):
        self.target_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)

    def create_spawn_layer(self):
        self.spawn_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)
    
    def create_agent_layer(self):
        self.agent_layer = OccupancyMap(self.x_min, self.x_max, self.y_min, self.y_max, self.voxel_size)

    def draw_box(self, layer, x_min, x_max, y_min, y_max):
        x_idx_min, y_idx_min = layer._get_voxel_coords(x_min, y_min)
        x_idx_max, y_idx_max = layer._get_voxel_coords(x_max, y_max)
        for x in range(x_idx_min, x_idx_max + 1):
            for y in range(y_idx_min, y_idx_max + 1):
                layer.bitmap[x, y] = True

    def draw_line(self, layer, x0, y0, x1, y1, resolution, thickness=0):
        xs = np.linspace(x0, x1, num=resolution)
        ys = np.linspace(y0, y1, num=resolution)
        for x, y in zip(xs, ys):
            layer.update(x, y, thickness=thickness)

    def update_agent(self, agent_x, agent_y):
        if self.last_pose is not None:
            x_old, y_old = self.last_pose
            self.draw_line(self.agent_layer, x_old, y_old, agent_x, agent_y)
            self.last_pose = [agent_x, agent_y]
        self.agent_layer.update(agent_x, agent_y, thickness=2)


    def update_target(self, target_x, target_y, target_radius=0.25):
        self.draw_box(self.target_layer,
                      target_x - target_radius,
                      target_x + target_radius,
                      target_y - target_radius,
                      target_y + target_radius)

    def update_spawn(self, spawn_x, spawn_y, spawn_radius=0.25):
        self.draw_box(self.spawn_layer,
                      spawn_x - spawn_radius,
                      spawn_x + spawn_radius,
                      spawn_y - spawn_radius,
                      spawn_y + spawn_radius)

    def clear_nonstatic_layers(self):
        self.last_pose = None
        self.clear_agent_layer()
        self.clear_target_layer()
        self.clear_spawn_layer()
        self.render_image = np.ones((self.map_layer.bitmap.shape[0], self.map_layer.bitmap.shape[1], 3), dtype=np.uint8) * 255
        self.draw_map()

    def clear_agent_layer(self):
        self.agent_layer.clear()

    def clear_target_layer(self):
        self.target_layer.clear()

    def clear_spawn_layer(self):
        self.spawn_layer.clear()

    def render(self):
        self.render_image[self.agent_layer.bitmap] = self.agent_color
        self.render_image[self.target_layer.bitmap] = self.target_color
        self.render_image[self.spawn_layer.bitmap] = self.spawn_color
        return imresize(self.render_image, (self.render_resolution, self.render_resolution, 3))

class HuskyCoordinateNavigateEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, start_locations=None, render_map=True):
        HuskyNavigateEnv.__init__(self, config, gpu_count)
        self.default_z = self.config["initial_pos"][2]
        self.target_mu = self.config["target_distance_mu"]
        self.target_sigma = self.config["target_distance_sigma"]
        self.locations = self.get_valid_locations(start_locations)
        self.n_locations = self.locations.shape[0]
        self.start_location = self.select_agent_location()
        self.config["initial_pos"] = [self.start_location[0], self.start_location[1], self.default_z]
        self.target_location = self.select_target()
        self.target_radius = 0.5
        self.render_map = render_map
        self.render_resolution = 256
        if render_map:
            mesh_file = '/home/bradleyemi/visual-cortex-parent/GibsonEnv/gibson/assets/dataset/Hanson/mesh_z_up.obj'
            self.map_renderer = NavigationMapRenderer(-11.5, 5.5, -6, 11, mesh_file, self.default_z, 0.1, render_resolution=self.render_resolution)
            #mesh_file = '/home/bradleyemi/visual-cortex-parent/GibsonEnv/gibson/assets/dataset/Beechwood/mesh_z_up.obj'
            #self.map_renderer = NavigationMapRenderer(-12, 3, -7.5, 7.5, mesh_file, self.default_z, 0.1, render_resolution=self.render_resolution)

        
    def get_valid_locations(self, start_locations):
        return np.loadtxt(start_locations, delimiter=',')
    
    def select_agent_location(self):
        index = np.random.choice(range(self.n_locations))
        return self.locations[index,:]
        
    def select_target(self):
        distances = [np.linalg.norm(d - self.start_location) for d in self.locations]
        desired_distance = np.random.normal(self.target_mu, self.target_sigma)
        distance_errors = [abs(d - desired_distance) for d in distances]
        index = np.argmin(distance_errors)
        # make sure the target isn't the start location
        if distances[index] == 0:
            index = np.argsort(distance_errors)[1]
        self.distance_to_target = distances[index]
        return self.locations[index,:]

    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        obs["target"] = self.calculate_target_observation()
        if self.render_map:
            self.map_renderer.update_agent(*self.robot.get_position()[:2])
            obs["map"] = self.map_renderer.render()
        else:
            obs["map"] = np.zeros((self.resolution, self.resolution, 3))
        info["distance"] = self.distance_to_target
        return obs, rew, done, info

    def calculate_target_observation(self):
        angle_to_target = np.arctan2(self.target_location[1] - self.robot.get_position()[1],
                                     self.target_location[0] - self.robot.get_position()[0])
        agent_angle_to_target = angle_to_target - self.robot.get_rpy()[2]
        agent_distance_to_target = np.linalg.norm(self.robot.get_position()[0:2] - self.target_location)
        return np.array([np.cos(agent_angle_to_target), np.sin(agent_angle_to_target), agent_distance_to_target])

    def _rewards(self, action=None, debugmode=False):
        # Alive
        alive = -0.05
        # Dense progress
        distance_old = self.distance_to_target
        self.distance_to_target = np.linalg.norm(self.target_location - self.robot.get_position()[:2])
        progress = distance_old - self.distance_to_target
        # Wall collision
        wall_collision_cost = self.get_wall_collision_cost()
        # Goal reward
        close_to_goal = 0.0
        if self._close_to_goal():
            close_to_goal = 20.0
        return [alive, progress, wall_collision_cost, close_to_goal]

    def get_wall_collision_cost(self):
        wall_contact = []
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        if len(wall_contact) > 0:
            return -0.25
        else:
            return 0.0

    def _termination(self, action=None, debugmode=False):
        z = self.robot.get_position()[2]
        z_initial = self.config["initial_pos"][2]
        dead = abs(z - z_initial) > 0.5
        done = dead or self.nframe >= self.config["episode_length"] or self._close_to_goal()
        return done


    def _close_to_goal(self):
        target_vector = self.target_location - self.robot.get_position()[:2]
        return np.linalg.norm(target_vector) < self.target_radius

    def _reset(self):
        self.start_location = self.select_agent_location()
        self.config["initial_pos"] = [self.start_location[0], self.start_location[1], self.default_z]
        self.target_location = self.select_target()
        obs = HuskyNavigateEnv._reset(self)
        obs["target"] = self.calculate_target_observation()
        if self.render_map:
            self.map_renderer.clear_nonstatic_layers()
            self.map_renderer.update_target(*self.target_location)
            self.map_renderer.update_spawn(*self.start_location)
            self.map_renderer.update_agent(*self.start_location)
            obs["map"] = self.map_renderer.render()
        else:
            obs["map"] = np.zeros((self.resolution, self.resolution, 3))
        return obs

    def render_map_rgb(self):
        if self.render_map:
            return self.map_renderer.render()
        else:
            return np.zeros((self.resolution, self.resolution, 3))


class HuskyVisualNavigateEnv(HuskyNavigateEnv):
    def __init__(self, config, gpu_count=0, texture=True, valid_locations=None):
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
        self.use_texture = False
        if texture:
            self.use_texture = True
            self.texture_path = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), "wood.jpg")
            self.load_texture(self.texture_path)

    def load_texture(self, texture_path):
        wood = imread(texture_path)
        size = self.config["resolution"]
        self.texture_image = skimage.transform.resize(wood[:, 42:208], (size,size))
        self.texture_points = np.array([[0.,0.,size,size],[size,0.,0.,size]]).T

    def get_valid_locations(self, valid_locations):
        return np.loadtxt(valid_locations, delimiter=',')

    def _close_to_goal(self):
        target_vector = np.array([self.target_x, self.target_y])
        distance = np.linalg.norm(self.robot.get_position()[0:2] - target_vector)
        return distance < self.target_radius

    def _rewards(self, action=None, debugmode=False):
        x, y, z = self.robot.get_position()
        z_initial = self.config["initial_pos"][2]
        roll, pitch = self.robot.get_rpy()[0:2]
        dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        dead_penalty = 0.0
        #if dead:
        #    dead_penalty = -10.0
        
        close_to_goal = 0.
        if self._close_to_goal():
            close_to_goal = 10.
        alive = -0.025
        return [alive, close_to_goal]
    
    def _step(self, action):
        obs, rew, done, info = HuskyNavigateEnv._step(self, action)
        self.depth = copy(obs["depth"]).squeeze()
        self.depth[self.depth == 0] = np.inf
        obs["rgb_filled"] = self._add_cube_into_image(obs, [self.target_x, self.target_y, self.default_z])
        return obs, rew, done, info

    def _add_cube_into_image(self, obs, location, color=[20,200,200]):
        cube = Cube(origin=np.array(location), scale=self.cube_size)
        self.cube_image = final = copy(obs["rgb_filled"])
        roll, pitch, yaw = self.robot.get_rpy()
        x, y, z = self.robot.eyes.get_position()
        size = self.config["resolution"] // 2
        fov = self.config["fov"]
        world_to_image_mat = generate_projection_matrix(x, y, z, yaw, pitch, roll, fov, fov, size, size)
        if self.use_texture:
            masks, xx_faces, yy_faces = get_cube_depth_and_faces(cube, world_to_image_mat, size*2, size*2)
            for face_mask, xx, yy in zip(masks, xx_faces, yy_faces):
                transform = skimage.transform.ProjectiveTransform()
                dest = np.array([yy,xx]).T
                try:
                    transform.estimate(self.texture_points, dest)
                    self.new_img = skimage.transform.warp(self.texture_image, transform.inverse)
                except:
                    continue
                img_mask = face_mask < self.depth
                self.cube_image[img_mask] = np.array(self.new_img[img_mask] * 255, dtype=np.uint8)
                self.depth[img_mask] = face_mask[img_mask]
        else:
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
        print("Distance to target:", distance)
        return [spawn_x, spawn_y, self.default_z]

    def _reset(self):
        self.target_x, self.target_y = self.select_new_target()
        self.config["initial_pos"] = self.select_new_agent_location()
        obs = HuskyNavigateEnv._reset(self)
        self.cube_image = copy(obs["rgb_filled"])
        self.depth = copy(obs["depth"]).squeeze()
        obs["rgb_filled"] = self._add_cube_into_image(obs, [self.target_x, self.target_y, self.default_z])
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
        #dead = abs(roll) > 1.22 or abs(pitch) > 1.22 or abs(z - z_initial) > 0.5
        done = out_of_bounds or self.nframe >= self.config["episode_length"] or self._close_to_goal()
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
