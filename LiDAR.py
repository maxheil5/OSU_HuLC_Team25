#Base Libraries
import numpy as np

# 3D Libraries
import open3d as o3d
import laspy

#Data Profiling
las= laspy.read("C:\\Users\\ryane\\Downloads\\486000_5455000\\486000_5455000.las")

print(np.unique(las.classification))
print([dimension.name for dimension in las.point_format.dimensions])
crs=las.vlrs[2].string
print(las.vlrs[2].string)

#Data Pre-Processing
pts_mask=las.classification == 6
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))
pcd_o3d=o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())

pcd_center = pcd_o3d.get_center()
pcd_o3d.translate(-pcd_center)

# Lines 39-42 create a black sphere that can be used to represent the lidar sensor
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
sphere.compute_vertex_normals()  # For rendering smoothness
sphere.paint_uniform_color([0.0, 0.0, 0.0])  # Color the sphere black
sphere.translate([0, 0, 500])

o3d.visualization.draw_geometries([pcd_o3d, sphere])  #displays the point cloud black sphere for visualization

#for loop to change the distance the sphere is at and measure the z values using the sphere as a reference point

sphere_center = np.array([0, 0, 500])

for z_position in range(500,99,-5): #changes the spheres position by starting from 500 and going down to 100
    sphere.translate([0,0,z_position - sphere.get_center()[2]])
    sphere_center[2] = z_position
    relative_z = xyz_t[2,:] - sphere_center[2]

    print(f"First 5 relative Z-values: {relative_z[:5]}\n") #gives first 5 points from point cloud to show the z-cord is being measured and changing
    #updates coordinates
    updated_xyz_t = xyz_t.copy()
    updated_xyz_t[2, :] = relative_z
    updated_pcd_o3d = o3d.geometry.PointCloud()
    updated_pcd_o3d.points = o3d.utility.Vector3dVector(updated_xyz_t.transpose())















