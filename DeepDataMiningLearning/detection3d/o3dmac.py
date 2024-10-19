#pip3 install open3d
#Open3D provides experimental support for 64-bit ARM architecture (arm64 or aarch64) on Linux and macOS (Apple Silicon). Starting from Open3D 0.14

# Test the legacy visualizer
#python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw_geometries([c])"

# Test the new GUI visualizer
#python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw(c)"

r"""
python -c "import open3d as o3d; \
           mesh = o3d.geometry.TriangleMesh.create_sphere(); \
           mesh.compute_vertex_normals(); \
           o3d.visualization.draw(mesh, raw_mode=True)"
"""