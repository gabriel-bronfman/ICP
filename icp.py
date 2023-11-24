import numpy as np
import logging
import copy
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def findNearestPoints(source, target):

    """
    param: source - a source point cloud
    param: target - your target point cloud

    return: an index-matched list of points that correspond between source and target closest matches
    """

    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(np.asarray(target.points))
    distances, indices = neigh.kneighbors(np.asarray(source.points), return_distance=True)

    distances = distances.ravel()
    indices = indices.ravel()

    points = np.zeros((len(np.asarray(source.points)),2,3))

    

    for index_point,index_match in enumerate(indices):
        points[index_point][0] = source.points[index_point]
        points[index_point][1] = target.points[index_match]

    
    return points, distances

def find_best_fit(source_points, target_points):

    assert source_points.shape == target_points.shape

    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    source_demean = source_points - centroid_source
    target_demean = target_points - centroid_target

    sigma = source_demean.T@target_demean


    U,S,V = np.linalg.svd(sigma)

    R = np.dot(V.T, U.T)

    if np.linalg.det(R) < 0:
       V[2,:] *= -1
       R = np.dot(V.T, U.T)

    t = centroid_target - R@centroid_source

    T = np.eye(4)

    T[:3,:3] = R
    T[:3,3] = t
    return T

def icp(source, target, initial_pose=None, max_iterations=70, accuracy=.00001):

    prev_err = 0

    if initial_pose is not None:
        source = source.transform(initial_pose)

    for i in range(max_iterations):
        
        points, distances = findNearestPoints(source, target)

        T = find_best_fit(points[:,0,:],points[:,1,:])

        source = source.transform(T)

        curr_err = np.mean(distances)
        logging.debug(f"Avg error: {curr_err}")
        # if np.abs(curr_err-prev_err) < accuracy:
        #     break
        # prev_err = curr_err
        
    
    return T, prev_err



def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
    zoom=0.4459,
    front=[0.9288, -0.2951, -0.2242],
    lookat=[1.6784, 2.0612, 1.4451],
    up=[-0.3402, -0.9189, -0.1996])


# Configure logging
logging.basicConfig(level=logging.DEBUG)


demo_icp_pcds = o3d.data.DemoICPPointClouds()
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
T = np.eye(4)

## Part 1
#For initial pose comparison
t = np.eye(4)

t,_ = icp(source, target, t)

draw_registration_result(source,target,t)


## Part two

target = o3d.io.read_point_cloud("kitti_frame1.pcd")
source = o3d.io.read_point_cloud("kitti_frame2.pcd")

t = np.eye(4)

t,_ = icp(source, target, t)

draw_registration_result(source,target,t)