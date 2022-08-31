import numpy as np
import cv2
import os
from pgd_gramm import *

# This module will render the scene generated by the rules define in the pgd_grammer.py  
class renderer:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self.canvas = np.ones((canvas_size, canvas_size, 3))*150.0 # Defining the image for drawing layouts and objects 
        self.canvas[:,:,1] = 250
        self.road_color = [200,200,250]
        self.sw_color = [50,50,70-20]
        self.object_colors = [[100,100,200], [50,200,50], [200,100,100], [200,200,100]] # | left, right, top, bottom 
        self.ped_color = [250, 250, 250] # pedestrian colors 
        self.build_color = [80, 80, 50] # building color for the background 
        self.x_threshold = int(canvas_size / 2)
        self.y_threshold = 0

    def render_layout(self, layout_param_properties):
        # Sample road layout parameters 
        # {'road_properties': {'width': 26, 'length': 124, 'n_lanes': 2, 'lane_width': 13.0, 'road_type': 'intersect', 'intersect_loc': 3}

        width = layout_param_properties['width']
        length = layout_param_properties['length']
        road_type = layout_param_properties['road_type']
        intersect_loc = layout_param_properties['intersect_loc']

        # Accessing the location of the roads based on the defined rules 
        main_road_start_location = layout_param_properties['start_main_road']
        main_road_end_location = layout_param_properties['end_main_road']
        i_road_start_location = layout_param_properties['start_i_road']
        i_road_end_location = layout_param_properties['end_i_road']

        # Location of the sidewalks in the scene
        sidewalks = layout_param_properties['sidewalks']
        n_sidewalks = layout_param_properties['n_sidewalks']

        # Locating the buildings in the background region to render realistc scenes 
        building_location = layout_param_properties['building_locs']
        n_buildings = layout_param_properties['n_buildings'] 

        for id in range(0, n_buildings):
            start_loc = (int(building_location[id][0][0] + self.x_threshold), int(building_location[id][0][1] + self.y_threshold))
            end_loc = (int(building_location[id][1][0] + self.x_threshold), int(building_location[id][1][1] + self.y_threshold))
            # print("building start loc: {}, end loc: {}".format(start_loc, end_loc))
            cv2.rectangle(self.canvas, start_loc, end_loc, color=self.build_color, thickness=-1) 

        for id in range(0, n_sidewalks):
            start_loc = (int(sidewalks[id][0][0] + self.x_threshold), int(sidewalks[id][0][1] + self.y_threshold))
            end_loc = (int(sidewalks[id][1][0] + self.x_threshold), int(sidewalks[id][1][1] + self.y_threshold))
            # print("start loc: {}, end loc: {}".format(start_loc, end_loc))
            cv2.rectangle(self.canvas, start_loc, end_loc, color=self.sw_color, thickness=-1)

        # Transforming the road coordinate system to camera coordinate system 
        mstart_loc = (main_road_start_location[0] + self.x_threshold, main_road_start_location[1] + self.y_threshold)
        mend_loc = (main_road_end_location[0] + self.x_threshold, main_road_end_location[1] + self.y_threshold)
        istart_loc = (i_road_start_location[0] + self.x_threshold, i_road_start_location[1] + self.y_threshold)
        iend_loc = (i_road_end_location[0] + self.x_threshold, i_road_end_location[1] + self.y_threshold)

        # Drawing the main road using the coordinates of the object 
        cv2.rectangle(self.canvas, mstart_loc, mend_loc, color=self.road_color, thickness=-1)

        if (road_type == 'intersect'):
            cv2.rectangle(self.canvas, istart_loc, iend_loc, color=self.road_color, thickness=-1)

        self.layout_map = self.canvas.copy()
        return self.canvas
    
    # Function will render the objecto on already formed canvas and using the color coding for the object orientation 
    def render_object(self, object_size, object_locs, lane):
        if (lane == 'left'):
            color_box = self.object_colors[0]
            start = (int(object_locs[0] - (object_size[0]//2)) + self.x_threshold, int(object_locs[1] - (object_size[1]//2) + self.y_threshold))
            end = (int(object_locs[0] + (object_size[0]//2)) + self.x_threshold, int(object_locs[1] + (object_size[1]//2) + self.y_threshold))    

            cv2.rectangle(self.canvas, start, end, color=color_box, thickness=-1)

        elif (lane == 'right'):
            color_box = self.object_colors[1]
            start = (int(object_locs[0] - (object_size[0]//2)) + self.x_threshold, int(object_locs[1] - (object_size[1]//2)) + self.y_threshold)
            end = (int(object_locs[0] + (object_size[0]//2)) + self.x_threshold, int(object_locs[1] + (object_size[1]//2)) + self.y_threshold)    

            cv2.rectangle(self.canvas, start, end, color=color_box, thickness=-1)

        elif (lane == 'bottom'):
            color_box = self.object_colors[2]
            start = (int(object_locs[0] - (object_size[1]//2)) + self.x_threshold, int(object_locs[1] - (object_size[0]//2)) + self.y_threshold)
            end = (int(object_locs[0] + (object_size[1]//2)) + self.x_threshold, int(object_locs[1] + (object_size[0]//2)) + self.y_threshold)    

            cv2.rectangle(self.canvas, start, end, color=color_box, thickness=-1)

        elif (lane == 'top'):
            color_box = self.object_colors[3]
            start = (int(object_locs[0] - (object_size[1]//2)) + self.x_threshold, int(object_locs[1] - (object_size[0]//2)) + self.y_threshold)
            end = (int(object_locs[0] + (object_size[1]//2)) + self.x_threshold, int(object_locs[1] + (object_size[0]//2)) + self.y_threshold)    

            cv2.rectangle(self.canvas, start, end, color=color_box, thickness=-1)
        
        elif (lane == 'sw'):
            color_box = self.ped_color
            start = (int(object_locs[0] - (object_size[1]//2)) + self.x_threshold, int(object_locs[1] - (object_size[1]//2)) + self.y_threshold)
            end = (int(object_locs[0] + (object_size[1]//2)) + self.x_threshold, int(object_locs[1] + (object_size[0]//2)) + self.y_threshold)

            circ_x = int(object_locs[0]) + self.x_threshold
            circ_y = int(object_locs[1]) + self.y_threshold
            rad = object_size[0] // 2

            cv2.circle(self.canvas, (circ_x, circ_y), radius = rad, color=color_box, thickness=-1)
            # cv2.rectangle(self.canvas, start, end, color=color_box, thickness=-1)

    # This function will render objects on a given lane based on the properties of the lane and the object location and size
    def render_lane_objects(self, object_locations_main):
        for lid in range(0, len(object_locations_main)):
            lane = list(object_locations_main[lid].keys())[0]
            object_locs = object_locations_main[lid][lane]

            # pdb.set_trace()
            # This will render each object in the list one by one by using the lane for color coding 
            for oid in range(0, object_locs.shape[0]):
                self.render_object(self.object_size, object_locs[oid, ...], lane) 

    def render_sw_objects(self, ped_locations):
        for swid in range(0, len(ped_locations)):
            self.render_object(self.ped_size, ped_locations[swid], 'sw')  # Just pasing a filler for using as lane name for rendering 

    def render_objects(self, object_params_properties):
        # Sample object properties at a given location
        # {'object_size': [10, 20], 'object_locations_main': [{'left': [[6.0, 48.0], [6.0, 72.0]]}, {'right': [[16.5, 0.0], [18.5, 48.0]]}], 
        # 'object_locations_i': [{'bottom': array([[  0., 143.], [ 72., 142.]])}, {'top': array([[ 24. , 152.5], [ 48. , 155.5]])}]}}

        self.object_size = object_params_properties['object_size']
        self.ped_size = object_params_properties['ped_size'] 
        self.object_locations_main = object_params_properties['object_locations_main']
        self.object_locations_i = object_params_properties['object_locations_i']
        self.ped_locations = object_params_properties['ped_locations'] 

        # print("Object properties .......")
        # print(object_params_properties)

        self.render_lane_objects(self.object_locations_main) # Rendering the objects on the main pathway 
        self.render_lane_objects(self.object_locations_i)  # Rendering the objects in the cross road
        self.render_sw_objects(self.ped_locations) # Rendering the pederstrains in the sidewalks 

        self.filled_layout = self.canvas
        return self.filled_layout

    def render_scene(self, layout_param_properties, object_param_properties, save_path):
        self.render_layout(layout_param_properties)
        self.render_objects(object_param_properties)

        buffer = np.ones((self.layout_map.shape[0], 10, 3)) * 255
        # combined_image = np.hstack([self.layout_map, buffer, self.filled_layout])

        # Currently saving just the final output of the rendering in an image
        combined_image = self.filled_layout
        flip_image = cv2.flip(combined_image, 0)
        # flip_image = cv2.flip(flip_image, )
        cv2.imwrite(save_path, flip_image)


# This function will create dataset with randomly selecting the parameters for the grammar generation 
def create_dataset():
    road_types = ['straight', 'intersect']
    canvas_size = 512
    object_size = [14,28]
    ped_size = [14,14] 

    n_lanes_list = [2,4,6]
    n_per_lane = 5
    n_ped_per_lane = 15
    save_path_root = './rendered_data/'

    object_densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    ped_densities = [0.0, 0.1, 0.2, 0.3, 0.4]

    n_samples = 10
    for id in range(0, n_samples):
        road_type = random.choice(road_types)
        n_lanes = random.choice(n_lanes_list)
        object_density = random.choice(object_densities)
        ped_density = random.choice(ped_densities)

        scene_layout = road_scene(road_type, object_size, n_lanes, n_per_lane, object_density, ped_density)
        scene_layout.fill_traffic(n_per_lane, n_ped_per_lane, ped_size)

        rendering_params = scene_layout.get_render_properties()
        print("Rendering scene: {}".format(id))
        rendering_engine = renderer(canvas_size)
        
        save_path = os.path.join(save_path_root, str(id) + '_render_img.png')
        rendering_engine.render_scene(rendering_params['road_properties'], rendering_params['object_properties'], save_path)



# Main function to test the functionality of each of the component 
def run_main(): 
    road_type = 'intersect'
    object_size = [10,20] 
    ped_size = [4,4]
    n_lanes = 4
    n_per_lane = 5
    n_ped_per_lane = 15


    n_tries = 10
    for id in range(0, n_tries):
        scene_layout = road_scene(road_type, object_size, n_lanes, n_per_lane, object_density=0.1, ped_density=0.2)
        scene_layout.fill_traffic(n_per_lane, n_ped_per_lane, ped_size)

        rendering_params = scene_layout.get_render_properties()
        # print("Rendering parameters ................") 
        # print(rendering_params['road_properties']['sidewalks'])

        print("Object properties: {}".format(rendering_params))

        print("Rendering scene: {}".format(id)) 
        canvas_size = 350
        rendering_engine = renderer(canvas_size)
        save_path = './debug/combined_rendered_{}_lanes_{}_road_type'.format(n_lanes, road_type) + str(id) + '.png'
        rendering_engine.render_scene(rendering_params['road_properties'], rendering_params['object_properties'], save_path)



if __name__ == "__main__":
    run_main()
    # create_dataset()