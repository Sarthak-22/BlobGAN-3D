from venv import create
from skspatial.objects import Plane
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math


# This function will rotate the given axis using the rotation matrix for the given shape
def rot_axis(axis, rot_mat, shift):
    rot_axis = [rot_mat.apply([a[0]-shift, a[1]-shift, a[2]-shift]) + shift for a in axis]
    return rot_axis


# This function will create a rotation of the given volume about three axis
def rotation_volume(volume, axises, x_angle, y_angle, z_angle):
    rot_volume = np.zeros_like(volume)

    # rot about [x, y, z, alpha] 
    rot_mat = R.from_quat([np.sin(x_angle/2), np.sin(y_angle/2), np.sin(z_angle/2), np.cos(np.pi/4)]) 
    print("rot matrix: {}".format(rot_mat.as_matrix()))

    vol_dim = volume.shape[0]
    shift = vol_dim // 2

    axises_rot = [rot_axis(axises[0], rot_mat, shift), rot_axis(axises[1], rot_mat, shift), rot_axis(axises[2], rot_mat, shift)]
    # print("Not rotating the axis currently ...")
    # axises_rot = axises

    for x in range(0, vol_dim):
        for y in range(0, vol_dim):
            for z in range(0, vol_dim):          
                x_val = x - shift
                y_val = y - shift
                z_val = z - shift

                # If the point is occupied, then rotate it using the rotation matrix 
                if (volume[x, y, z] == 1):
                    point_loc = [x_val, y_val, z_val]
                    rot_loc = rot_mat.apply(point_loc) + shift
                    rot_volume[int(rot_loc[0]), int(rot_loc[1]), int(rot_loc[2])] = 1

    return rot_volume, axises_rot


# This function will create a 3D ellipse using the given axis and the volume dimension for each side of the cube
def create_3d_ellipse(a, b, c, K, vol_dim):
    volume = np.zeros((vol_dim, vol_dim, vol_dim))
    center_x, center_y, center_z = vol_dim//2, vol_dim//2, vol_dim//2

    for x in range(-a, a):
        for y in range(-a, a):
            for z in range(-a, a):
                sdf_val = (x * x)/(a*a) + (y * y)/(b*b) + (z * z)/(c*c) - K
                if (sdf_val < 0):
                    volume[center_x + x, center_y + y, center_z + z] = 1

    return volume

# This function will render the given axis on the plane 
def render_axis(plane, axis):
    projected_points = []
    # print("axis : {}".format(axis))
    for k in axis: 
        k_val = k
        point = plane.project_point(k)
        projected_points.append(point) 
    rend_axis = np.array(projected_points)
    # print("render axis shape: {}".format(rend_axis.shape))
    return np.array(projected_points)

# This will render all the aixes togather 
def render_all_axis(axises, plane):
    axisa = render_axis(plane, axises[0])
    axisb = render_axis(plane, axises[1])
    axisc = render_axis(plane, axises[2])

    # print("axises shape: {} {} {}".format(axisa.shape, axisb.shape, axisc.shape))
    return [axisa, axisb, axisc]

# This function will render the volume along with the main axises at a given plane using sklearn 
def render_volume(volume, plane):
    # Accumulator array to accumulate the projected points onto a given plane 
    projected_points = []
    for a in range(0, volume.shape[0]):
        for b in range(0, volume.shape[1]):
            for c in range(0, volume.shape[2]):
                if (volume[a, b, c] == 1):
                    point = plane.project_point([a,b,c])
                    projected_points.append(point)

    return np.array(projected_points)

# This function will create a graph for displaying the ellipsoid and its projection on multiple planes
def process_display(projection_set, projection_axises, save_name, prec_factor, col):
    # Projectiong of x, hence ignoring the first index of the plot
    projection_set = np.concatenate((projection_set, [[0,0], [prec_factor,prec_factor]]), axis=0)
    print("projected points shape: {}".format(projection_set.shape)) 
    plt.scatter(projection_set[:,0], projection_set[:,1], c=col, s=0.4)

    # Rendering the axises 
    colors = ['black', 'purple', 'navy']
    print("colors: {}".format(colors))
    for aid in range(0,3):
        col = colors[aid]
        plt.scatter(projection_axises[aid][:,0], projection_axises[aid][:,1], s=5.0, c=col, alpha=0.8)            

    plt.title(save_name[7:-4])
    plt.savefig(save_name)
    plt.clf()

# This is the main function that we will use for processing 
def run_main():
    prec_factor = 100
    a = prec_factor // 2
    b = prec_factor // 4
    c = prec_factor // 8
    K = 1.0

    vol_dim = prec_factor
    print("Creating 3D ellipse in the volume of shape: {}".format(vol_dim))
    volume = create_3d_ellipse(a,b,c,K,vol_dim) 
    axises = []
    
    shift = vol_dim // 2
    aaxis = [[id + shift, 0 + shift, 0 + shift] for id in range(-a, a)]
    baxis = [[0 + shift, id + shift, 0 + shift] for id in range(-b, b)]
    caxis = [[0 + shift, 0 + shift, id + shift] for id in range(-c, c)]
    # print("axis : {}".format(caxis))
    
    # Major and minor axises stored in a a list 
    axises.append(aaxis)
    axises.append(baxis)
    axises.append(caxis)

    volume, rot_axis = rotation_volume(volume, axises, 0, np.pi/4, np.pi/4)
    # volume, rot_axis = rotation_volume(volume, axises, 0, 0, 0)

    plane_x = Plane(point=[-prec_factor, 0, 0], normal=[1, 0, 0])
    plane_y = Plane(point=[0, -prec_factor, 0], normal=[0, 1, 0])  
    plane_z = Plane(point=[0, 0, -prec_factor], normal=[0, 0, 1])  

    # Projecting on one of the planes 
    projection_x = render_volume(volume, plane_x)[:,1:]    
    projection_axises = render_all_axis(rot_axis, plane_x) 
    projection_axises = [pa[:,1:] for pa in projection_axises]

    print("Projecting the input on the plane x ...")
    process_display(projection_x, projection_axises, './figs/x_proj.png', prec_factor, 'lightcoral')

    # Projection on second y plane
    projection_y = render_volume(volume, plane_y)
    projection_y = np.concatenate((projection_y[:,:1], projection_y[:,2:]), axis=1)
    projection_axises = render_all_axis(rot_axis, plane_y)
    projection_axises = [np.concatenate((pa[:,:1], pa[:,2:]), axis=1) for pa in projection_axises] 
    
    print("Projection the input on the plane y ...")
    process_display(projection_y, projection_axises, './figs/y_proj.png', prec_factor, 'y')


    projection_z = render_volume(volume, plane_z)[:,:2]
    projection_z = np.concatenate((projection_z, [[0,0], [prec_factor,prec_factor]]), axis=0)
    projection_axises = render_all_axis(rot_axis, plane_z) 
    projection_axises = [pa[:,:2] for pa in projection_axises]

    print("Projecting the input volume on the plane z ...") 
    process_display(projection_z, projection_axises, './figs/z_proj.png', prec_factor, 'lavender')
    
    
if __name__ == "__main__":
    print("Performing the main operation")
    run_main()
 
