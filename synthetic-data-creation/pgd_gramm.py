import numpy as np
import random 
import pdb

class lane:
    def __init__(self, width, length, object_density):
        # Convention: x-coord - width, y-coord - length
        self.width = width
        self.length = length
        self.buffer = 0.5 # fraction of object size to make space between two objects 
        self.w_shift = 0.2 # fraction of the width by which the objects will be shifted along the x axis 
        self.drop_chance = 1 - object_density # The drop chances are inversly correlated with the required object density
        # pdb.set_trace()

    def get_shape(self):
        return self.width, self.length 
    
    def get_orientation(self):
        return self.orientation, self.side

    # This function will spawn points on a given lane based on the object size and the number of objects 
    def spawn_points(self, object_size, n):
        self.object_w = object_size[0] # height of the object to be placed
        self.object_h = object_size[1] # width of the object to be placed 

        n_possible = int(self.length // (self.object_h*(1 + self.buffer))) # Max possible objects that can be present on the road 
        
        start_point = random.randint(int(self.object_h // 2), self.object_h) # Adding random noise of upto a size of a single object length to add the stochasticity 
        object_locations = []
        n_cp = 0 # Number of objects currently placed in the scene
        for id in range(0, n_possible):
            if (n_cp == n): # If we have already placed n objects then break the loop 
                break
            
            h_center = start_point + id * (self.object_h + self.object_h * self.buffer)  # Selecting the y coordinate
            
            # Randomly selecting a small region to shift the object withing that in x-direction 
            random_shift = random.uniform(-self.w_shift* self.object_w, self.w_shift* self.object_w)
            # random_shift = 0
            # Currently making the shift operation to be zero 

            w_center = int((self.width // 2) + random_shift) # Selecting the x coordinate | Perturbing the x-center using some random value of fraction of object width 

            dropping = np.random.binomial(1, self.drop_chance)
            if (not dropping == 1):
                location = [w_center, h_center]
                object_locations.append(location)
                n_cp += 1 

        # Returning the object locations 
        success = n_cp > 1
        return success, np.array(object_locations)

class road:
    def __init__(self, width, length, n_lanes, object_size, road_type, object_density, ped_density):
        self.width = width
        self.length = length 
        self.n_lanes = n_lanes
        self.object_size = object_size # Size of the object to be placed at a given location  
        self.max_object_dim = max(object_size[0], object_size[1]) # This will take the maximum dimension of the object for removing the possibility of intersections. 
        self.lane_width = width / n_lanes
        self.road_type = road_type
        self.traffic_flow_dir = 0.4
        self.ped_density = ped_density
        self.object_density = object_density

        self.sw_size = self.lane_width // 2 # Side walk size is defined as the half of the lane size 
        # pdb.set_trace()

        # This will create a main road, where objects will be placed. 
        self.lanes = [] # | To store the lane object for each entity in the road 
        self.sidewalks = [] # | Similar to the list of lanes, this will have the list of side-walks for the scene, similar to road we will also spawn points \
                             # to locate the pedestrians 

        # Defining the point of interesection on the road 
        self.intersect_loc = random.randint(0,self.length - self.width //2) # | Indexing by the center location for the road which will be used for shifting objects 

        # Initializing the side-walks, as the sidewalks do not have the orientation, they can be stored in a single list of lane objects 
        if (self.road_type == 'straight'):
            for id in range(0,2): # | Adding only the two sidewalks for the road one on the each side 
                cur_lane = lane(self.sw_size, self.length, self.ped_density)
                self.sidewalks.append(cur_lane)                              
            self.n_sidewalks = 2  
        
        elif (self.road_type == 'intersect'):
            #            7|  |8
            #             |  |
            # ------1-----    -----3-----
            # ------2-----    -----4-----
            #            5|  |6
            #             |  |

            # Firstly taking the four sidewalks which will have the same length 
            for id in range(0,4):
                sw_length = (self.length - self.width) // 2
                cur_lane = lane(self.sw_size, sw_length, self.ped_density) 
                self.sidewalks.append(cur_lane)

            # Taking the lane 5 and 6 which will have the same length 
            for id in range(4, 6):
                sw_length = self.intersect_loc
                cur_lane = lane(self.sw_size, sw_length, self.ped_density)
                self.sidewalks.append(cur_lane)

            for id in range(6,8):
                sw_length = self.length - self.width - self.intersect_loc 
                cur_lane = lane(self.sw_size, sw_length, self.ped_density)
                self.sidewalks.append(cur_lane)
            
            self.n_sidewalks = 8 
            # pdb.set_trace()
        
        # Left and right lanes will be common to both the road types, straight and the intersect one 
        self.sides = ['left']* (self.n_lanes // 2) + ['right']* (self.n_lanes // 2) # | This will generate a sequence of lanes with left right left right upto number of lanes 
        # Iterating over all the lanes and instantiating object of each lane separately 
        for id in range(0, n_lanes):
            current_lane = lane(self.lane_width, self.length, self.object_density)
            self.lanes.append(current_lane)

        # This will create a intersection over the given road which will be horizontal 90 degree intersection 
        if (self.road_type == 'intersect'):
            self.i_sides = ['bottom'] * (self.n_lanes//2) + ['top'] * (self.n_lanes // 2) # | This will generate a sequence of lanes left and right on the horizontal road 
            self.i_lanes = [] # | Sidelanes on the intersection roads defined 

            for id in range(0, n_lanes):
                current_lane = lane(self.lane_width, self.length, self.object_density)
                self.i_lanes.append(current_lane) 


        # Initialize the sidewalks locations, so that it can be used directly later 
        self.sidewalks_locs = self.get_sidewalks_locs()

        # Initialize the bulding locations, so that we can use them for rendering 
        self.building_locs = self.get_building_locs()
        self.n_buildings = len(self.building_locs)
            
    # This function will return the location for the sidewalks, Note that there is no orientation for the sidewalks, 
    # we can just return the start and the end location for the box 
    def get_sidewalks_locs(self):
        sidewalks_locs = []
        if (self.road_type == 'straight'): # | For the straight road only two side walks will be present
            #     |  |
            #    1|  |2
            #     |  |
            #     |  |

            sw1_loc_start = (-self.sw_size, 0)
            sw1_loc_end = (0, self.length)
            sidewalks_locs.append([sw1_loc_start, sw1_loc_end])

            sw2_loc_start = (self.width, 0)
            sw2_loc_end = (self.width + self.sw_size, self.length)
            sidewalks_locs.append([sw2_loc_start, sw2_loc_end])

        elif(self.road_type == 'intersect'): # | For the interesection road eight sidewalks will be present 
            #            7|  |8
            #             |  |
            # ------1-----    -----3-----
            # ------2-----    -----4-----
            #            5|  |6
            #             |  |

            # [start_loc, end_loc]
            sw1_locs = [(-int(self.length/2 - self.width/2), self.intersect_loc+self.width), (0, self.intersect_loc+self.width+self.sw_size)]
            sw2_locs = [(-int(self.length/2 - self.width/2), self.intersect_loc-self.sw_size), (0, self.intersect_loc)]

            sw3_locs = [(self.width, self.intersect_loc+self.width), (int(self.length/2+self.width/2), self.intersect_loc+self.width+self.sw_size)]
            sw4_locs = [(self.width, self.intersect_loc-self.sw_size), (int(self.length/2+self.width/2), self.intersect_loc)]

            sw5_locs = [(-self.sw_size, 0), (0, self.intersect_loc)]
            sw6_locs = [(self.width, 0), (self.width+self.sw_size, self.intersect_loc)]

            sw7_locs = [(-self.sw_size, self.intersect_loc+self.width), (0, self.length)]
            sw8_locs = [(self.width, self.intersect_loc+self.width), (self.width+self.sw_size, self.length)]

            sidewalks_locs = [sw1_locs, sw2_locs, sw3_locs, sw4_locs, sw5_locs, sw6_locs, sw7_locs, sw8_locs]

        # Returning the sidewalks locations in both the cases defined based on the road types 
        return sidewalks_locs

    # This function will create the boxes for the grass background in the scene to distinguish sky from building 
    def get_building_locs(self):
        building_locs = []
        if (self.road_type == 'straight'):
            #        |
            #    bd1 |  bd2
            #        |

            bd1_start = (-int(self.length/2 - self.width/2), 0)
            bd1_end = (-self.sw_size, self.length)
            bd1 = [bd1_start, bd1_end]

            bd2_start = (self.width+self.sw_size,0)
            bd2_end = (int(self.length/2 + self.width/2), self.length)
            bd2 = [bd2_start, bd2_end]

            building_locs = [bd1, bd2]

        if (self.road_type == 'intersect'):
            #        |
            #    bd3 | bd4 
            # ------- -------
            #    bd1 | bd2
            #        |

            bd1_start = (-int(self.length/2 - self.width/2), 0)
            bd1_end = (-self.sw_size, self.intersect_loc - self.sw_size)
            bd1 = [bd1_start, bd1_end]

            bd2_start = (self.width + self.sw_size, 0)
            bd2_end = (int(self.length/2 + self.width/2), self.intersect_loc - self.sw_size)
            bd2 = [bd2_start, bd2_end]

            bd3_start = (-int(self.length/2-self.width/2), self.intersect_loc + self.width + self.sw_size)
            bd3_end = (-self.sw_size, self.length)
            bd3 = [bd3_start, bd3_end]

            bd4_start = (self.width + self.sw_size, self.intersect_loc + self.width + self.sw_size)
            bd4_end = (int(self.length/2 + self.width/2), self.length)
            bd4 = [bd4_start, bd4_end]

            building_locs = [bd1, bd2, bd3, bd4]

        return building_locs
            

    # This function will remove the objects from the given list based on the intersection regions provided     
    def remove_intersecting_objects(self, object_dict, intersect_region_x, intersect_region_y):
        filtered_object_list = {}
        lane_type = list(object_dict.keys())[0] # | As this dictionary has only a single key value which is for the lane object 
        object_list = object_dict[lane_type] # Accessing the object list by using the key value for the given object 
        # pdb.set_trace()

        object_list_filtered = []
        buffer = self.max_object_dim // 2 # This will be buffer, in which region the object center should not be present in the border region of the intersection 
        for id in range(0, object_list.shape[0]):
            x_condition = (object_list[id, 0] > intersect_region_x[0] - buffer and object_list[id, 0] < intersect_region_x[1] + buffer)
            y_condition = (object_list[id, 1] > intersect_region_y[0] - buffer and object_list[id, 1] < intersect_region_y[1] + buffer)

            # If the object is not in the intersection we can add it to the filtered list for the processing
            if (not (x_condition and y_condition)):
                obj_loc = [object_list[id,0], object_list[id,1]]
                object_list_filtered.append(obj_loc)

        # filling the values in the new dictionary for the objects 
        filtered_object_list[lane_type] = np.array(object_list_filtered)
        return filtered_object_list  

    # This function will spawn the pedestrains for all the sidewalks and transforms them to the global coordinate system for further rendering 
    def spawn_pedestrains(self, n_ped_per_lane, ped_size):
        combined_peds = []
        self.n_ped_per_lane = n_ped_per_lane
        self.ped_size = ped_size
        
        if (self.n_sidewalks == 2): # | The case without intersection 
            for id in range(0, self.n_sidewalks):
                # Placing objects in the selected lane 
                success, lane_objects = self.sidewalks[id].spawn_points(self.ped_size, self.n_ped_per_lane)

                if (success):
                    x_shift = self.sidewalks_locs[id][0][0] # Only taking the start of the box for obtaining the shift value 
                    y_shift = self.sidewalks_locs[id][0][1] # As the objects are defined from the bottom left corner of the scene 

                    lane_objects[:,0] += x_shift     
                    lane_objects[:,1] += y_shift # | IT will be zero 
                    
                    combined_peds.append(lane_objects)

        if (self.n_sidewalks == 8): # | The case with intersection, in this setup we can add objects directly in the four lanes other, for others we need to transpose    
            for id in range(0, self.n_sidewalks):
                success, lane_objects = self.sidewalks[id].spawn_points(ped_size, self.n_ped_per_lane)

                if (success):     
                    # IF the sidewalks are present aside the road going to the front 
                    if (id == 4 or id == 5 or id == 6 or id == 7):
                        x_shift = self.sidewalks_locs[id][0][0]
                        y_shift = self.sidewalks_locs[id][0][1]

                        lane_objects[:,0] += x_shift
                        lane_objects[:,1] += y_shift # | IT will be either zero or self.intersect_loc + self.width 

                        combined_peds.append(lane_objects)

                    # If the sidewalks are present aside the intersect road 
                    elif (id == 0 or id == 1 or id == 2 or id == 3):
                        lane_objects_transformed = np.zeros(lane_objects.shape)
                        lane_objects_transformed[:,0] = lane_objects[:,1] # Step 2 | Swapping the coordinates
                        lane_objects_transformed[:,1] = lane_objects[:,0] # Step 2 | Swapping the coordinates 

                        x_shift = self.sidewalks_locs[id][0][0]
                        y_shift = self.sidewalks_locs[id][0][1] 

                        lane_objects_transformed[:,0] += x_shift
                        lane_objects_transformed[:,1] += y_shift 

                        combined_peds.append(lane_objects_transformed)
        
        # Returning the combined set for all the pedestrians in the given sidewalks 
        # self.combined_peds = combined_peds
        if (len(combined_peds) > 0):
            self.combined_peds = np.concatenate(combined_peds, axis=0)
        else:
            self.combined_peds = combined_peds 
        return self.combined_peds, True  


    # This function will spawn the cars on the road by placing objects one by one on all the lanes individually 
    def spawn_cars_on_roads(self, n_per_lane):
        # As we have multiple lanes, we need to adjust location of cars from each lane to the road coordinate system,
        # Essential we have to perform delta shift over the x coordinated of the objects from each lane to obtain the correct
        # location with respect to the global road 

        self.combined_objects = [] # | List of objects of type dict:  (lane_type: array_of_object_locs) 
        self.combined_i_objects = [] # | List of objects of type dict:  (lane_type: array_of_object_locs) 
        # If there are no objects in the given lane, then that entry will not be added into the above lists 

        self.n_per_lane = n_per_lane # Number of ojects in a given lane 

        # Spawning points in the main road layout 
        for id in range(0, self.n_lanes):
            success, lane_objects = self.lanes[id].spawn_points(self.object_size, self.n_per_lane)
            
            # Defining a new way to return empty list 
            lane_side = self.sides[id]
            object_bundle = {lane_side: []}

            # If there are objects present in the lane then only we will perform the postprocessing
            if (success):
                # pdb.set_trace()
                dc_offset = self.lane_width * id # To account for multiple lanes in the road

                # Transformation matrix | Adjusting for the shift from local to global coordinates 
                lane_objects[:,0] += dc_offset 
                object_bundle[lane_side] = lane_objects 
            
                # Append the object bundle without any condition, even if it is empty 
                self.combined_objects.append(object_bundle)

        # Spawning points in the intersection or orthogonal road 
        if (self.road_type == 'intersect'):
            self.combined_i_objects = []
            for id in range(0, self.n_lanes):
                success, lane_objects = self.lanes[id].spawn_points(self.object_size, self.n_per_lane)
                
                # Defining a new way to return empty list in the dictionary 
                lane_side = self.i_sides[id]
                object_bundle = {lane_side: []} 

                # If there are objects present in the lane, then only we will perform post-processing over it
                if (success):
                    dc_offset = self.lane_width * id # To account for multiple lanes on the road

                    # pdb.set_trace()
                    # Transformation matrix | Adjusting for the shift from local to global coordinates. Followed by in-place rotation along the z-axis  
                    # Transformation operations
                    #   1. Add a dc offset in x direction to handle multiple lanes in a given road
                    #   2. Transpose the coordinates i.e. x->y and y->x 
                    #   3. Adding a dc shift of intersect location to shift the intersect road in the y-direction
                    #   4. Now, we have to make a shift towards left (-x direction) to align the center of intersect road
                    #      with the main road. The shift value will be : (length/2 - (width/2)) 

                    lane_objects[:,0] += dc_offset      # Step 1 
                    lane_objects_transformed = np.zeros(lane_objects.shape)
                    lane_objects_transformed[:,0] = lane_objects[:,1] # Step 2 | Swapping the coordinates
                    lane_objects_transformed[:,1] = lane_objects[:,0] # Step 2 | Swapping the coordinates 

                    lane_objects_transformed[:,1] += self.intersect_loc # Step 3 | shifting the lanes by a fixed dc value 
                    lane_objects_transformed[:,0] -= int((self.length / 2) - (self.width / 2)) # Step 4 | Adjusting the x-coordinate by shifting in negative x-direction 

                    object_bundle[lane_side] = lane_objects_transformed

                    # Add the object bundle in the list even if there are no objects spawned. 
                    self.combined_i_objects.append(object_bundle) 
    
        # If the road type is intersection, we have to filter out the cases where the two objects could interesect
        # Basically, we will have a single directional flow of traffic on the interesection and the ties will be broken arbitrarily 
        if (self.road_type == 'intersect'):
            intersect_region_y_min = self.intersect_loc # | starting point of the intersection in the y direction 
            intersect_region_y_max = self.intersect_loc + self.width # | end point of the intersection in the y direction 
            
            intersect_region_x = [0, self.width]
            intersect_region_y = [intersect_region_y_min, intersect_region_y_max]

            # print("Intersect region-x: {}".format(intersect_region_x))
            # print("Intersect region-y: {}".format(intersect_region_y))
            # set_trace()

            # We will remove the traffic from one of the roads based on a toss 
            main_flow = np.random.binomial(1, self.traffic_flow_dir) # Randomly selecting which flow to keep, either main flow or the orthogonal flow of object 
            if (main_flow):
                self.combined_i_objects_filtered = [] # | Filtered list of objects that will be kept and intersecting objects will be thrown away 
                for id in range(0, len(self.combined_i_objects)): # Iterating over all the lanes | # If there are no objects in the lane then, we dont need the filtering for post processing 
                    object_list = self.combined_i_objects[id] # Selecting the list of objects in the current list 
                    object_list_filtered = self.remove_intersecting_objects(object_list, intersect_region_x, intersect_region_y)
                    
                    # pdb.set_trace()
                    
                    self.combined_i_objects_filtered.append(object_list_filtered)
                    # pdb.set_trace()
                # Saving the replaced objects in the same container 
                self.combined_i_objects = self.combined_i_objects_filtered

            else: # If we are passing the intersection traffic 
                self.combined_objects_filtered = [] # | Filtered list of the objects that will be kept after removing interesection 
                for id in range(0, len(self.combined_objects)): # If there are no objects in the lane then we don't need the postprocessing filtering
                    object_list = self.combined_objects[id] 
                    object_list_filtered = self.remove_intersecting_objects(object_list, intersect_region_x, intersect_region_y)
                    self.combined_objects_filtered.append(object_list_filtered) 
                    # pdb.set_trace()

                # Saving the replaces objects in the same objects 
                self.combined_objects = self.combined_objects_filtered
        
        if (self.road_type == 'intersect'):
            return self.combined_objects, self.combined_i_objects
        else: # If we don't have interesection return None 
            return self.combined_objects, None 


    # This function will return all the required information for rendering of the road layout and the objects over it. 
    def get_rendering_params(self):
        start_location_main_road = (0,0)
        end_location_main_road = (self.width, self.length)

        start_location_i_road = (-int((self.length/2) - (self.width/2)), self.intersect_loc) 
        end_location_i_road = (int((self.length/2) + (self.width/2)), self.intersect_loc + self.width)   

        road_properties = {'width': self.width,
                           'length': self.length,
                           'n_lanes': self.n_lanes,
                           'lane_width': self.lane_width,
                           'road_type': self.road_type,
                           'intersect_loc': self.intersect_loc,
                           'start_main_road': start_location_main_road,
                           'end_main_road': end_location_main_road,
                           'start_i_road': start_location_i_road,
                           'end_i_road': end_location_i_road,
                           'n_sidewalks': self.n_sidewalks,
                           'sidewalks': self.sidewalks_locs,
                           'n_buildings': self.n_buildings,
                           'building_locs': self.building_locs, 
                           }  
        
        object_properties = {'object_size': self.object_size,
                            'object_locations_main': self.combined_objects,
                            'object_locations_i': self.combined_i_objects,
                            'ped_size': self.ped_size, 
                            'ped_locations': self.combined_peds} 

        output_bundle = {'road_properties': road_properties, 'object_properties': object_properties} 
        return output_bundle
                        


# This class will define a road scene,by creating layout and additional things over the layout 
class road_scene():
    def __init__(self, road_type, object_size, n_lanes, n_per_lane, object_density=0.5, ped_density=0.4):
        self.road_type = road_type
        self.object_size = object_size

        # For now we will define the width and the length of the road based on the given object size and randomly sampling from it 
        self.width = random.randint(int(1.5*object_size[0]*n_lanes), 2.0*object_size[0]*n_lanes)
        self.length = random.randint(16*object_size[1], 20*object_size[1])

        self.n_lanes = n_lanes
        self.n_per_lanes = n_per_lane

        # Defining the road layout with the given set of parameters
        self.road_layout = road(self.width, self.length, n_lanes, object_size, road_type, object_density, ped_density)

    # This function will spawn the cars on the roads using the defined algorithmic logic 
    def fill_traffic(self, n_per_lane, n_ped_per_lane, ped_size):
        self.road_layout.spawn_cars_on_roads(n_per_lane)
        self.road_layout.spawn_pedestrains(n_ped_per_lane, ped_size)

    # This function will return the rendering parameters for the road layout as well as the objects in the road 
    def get_render_properties(self):
        rendering_params = self.road_layout.get_rendering_params()
        return rendering_params 


# Main function to test the functionality of each of the component 
def run_main():
    # road_type = 'straight' 
    road_type = 'intersect'
    object_size = [10,20]
    ped_size = [4,4]
    n_lanes = 2
    n_per_lane = 4
    n_ped_per_lane = 4

    scene_layout = road_scene(road_type, object_size, n_lanes, n_per_lane, object_density=0.1, ped_density=0.2)
    scene_layout.fill_traffic(n_per_lane, n_ped_per_lane, ped_size)  

    rendering_params = scene_layout.get_render_properties()
    print("Rendering parameters ................") 
    print(rendering_params)


if __name__ == "__main__":
    run_main()

