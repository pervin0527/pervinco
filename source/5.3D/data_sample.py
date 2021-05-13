import os
import wget
import trimesh
from matplotlib import pyplot as plt

if not os.path.isfile('./cow.obj'):
    url = 'https://storage.googleapis.com/tensorflow-graphics/notebooks/index/cow.obj'
    wget.download(url)

mesh = trimesh.load('cow.obj')
print(mesh)

vertices = mesh.vertices
faces = mesh.faces

print(vertices.shape, faces.shape)

points = vertices

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()