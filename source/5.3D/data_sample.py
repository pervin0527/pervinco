import os
import wget
import trimesh
import numpy as np
from matplotlib import pyplot as plt

def show_point_cloud(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()

""" Understanding mesh data """
# trimesh.util.attach_to_log()

sample = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], 
                                   [0, 1, 1], [1, 0, 1], [1, 1, 1]],
                         faces=[[0, 1, 2], [0, 1, 5]],
                         process=False)
print(sample)
sample.show()

""" Convert mesh -> point cloud """
if not os.path.isfile('./cow.obj'):
    url = 'https://storage.googleapis.com/tensorflow-graphics/notebooks/index/cow.obj'
    wget.download(url)

mesh = trimesh.load('cow.obj')
vertices, faces = mesh.vertices, mesh.faces
print(vertices.shape, faces.shape)
show_point_cloud(vertices)

""" using sample .off file """
keras_mesh = trimesh.load('cow.obj')
keras_mesh.show()

keras_points, keras_faces = keras_mesh.vertices, keras_mesh.faces
sample_faces = keras_faces.flatten()
sample_faces = np.random.choice(sample_faces, 10, replace=False)
sample_faces = np.ravel(sample_faces, order='C')
print(sample_faces.shape)

sample_points = []
for idx in sample_faces:
    sample_points.append(keras_points[idx])

sample_points = np.array(sample_points)
print(sample_points.shape)

show_point_cloud(sample_points)