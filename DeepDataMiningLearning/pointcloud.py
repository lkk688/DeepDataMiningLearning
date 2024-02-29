import numpy as np
import open3d as o3d #pip3 install open3d
print(o3d.__version__)
from tqdm import tqdm
import matplotlib.pyplot as plt

from DeepDataMiningLearning.visionutil import read_image
r"""
python -c "import open3d as o3d; \
           mesh = o3d.geometry.TriangleMesh.create_sphere(); \
           mesh.compute_vertex_normals(); \
           o3d.visualization.draw(mesh, raw_mode=True)"
"""
#https://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html#from-numpy-to-open3d-image


class MyPointCloud(object):
    def __init__(self, file):
        img = self.read_image(file) #(427, 640) HW format uint8
        #img = img.transpose(1, 0) #(640, 427) WH format
        self.array = self.__preprocess(img) #(n,3)
        self.cloud = self.create_pcd(self.array)
    
    def read_image(self, path):
        return read_image(path, rgb=False)
    
    @staticmethod
    def __preprocess(array) -> dict:
        rgb = []
        # x, y = array.shape #H=427, W=640
        # avg = lambda d: sum(d) / len(d)
        # scale_factor = avg(array.shape)/array.max() #2.09
        # for row in tqdm(range(y)): #height 427
        #     for col in range(x): #width 640
        #         mod_depth = array[row][col]*scale_factor
        #         xyz = np.array([row, col, mod_depth])
        #         rgb.append(xyz)
        H, W = array.shape #H=427, W=640
        avg = lambda d: sum(d) / len(d)
        scale_factor = avg(array.shape)/array.max() #2.09
        for row in tqdm(range(H)): #height 427
            for col in range(W): #width 640
                mod_depth = array[row][col]*scale_factor
                xyz = np.array([row, col, mod_depth])
                rgb.append(xyz)

        rgb = np.array(rgb) #(273280, 3)
        print(f'Converted raw array of shape {array.shape} to RGB array of shape {rgb.shape}')
        return rgb 
    
    @staticmethod
    def create_pcd(array):
        pcd = o3d.geometry.PointCloud()
        ## From numpy to Open3D
        pcd.points = o3d.utility.Vector3dVector(array)
        return pcd
    
    @staticmethod
    def to_numpy(pcd):
        # From Open3D to numpy
        ## convert Open3D.o3d.geometry.PointCloud to numpy array
        np_points = np.asarray(pcd.points)#self.pcd.points)
        return np_points
    
    @staticmethod
    def saveply(pcd, outpath="data/pointcloud.ply"):
        o3d.io.write_point_cloud(outpath, pcd)
    
    @staticmethod
    def loadply(path="data/pointcloud.ply"):
        pcd = o3d.io.read_point_cloud(path)
        return pcd
    
    @staticmethod
    def gen_simdata():
        # generate some neat n times 3 matrix using a variant of sync function
        x = np.linspace(-3, 3, 401)
        mesh_x, mesh_y = np.meshgrid(x, x) #mesh_x: (401, 401)
        z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2))) #(401, 401)
        z_norm = (z - z.min()) / (z.max() - z.min()) #(401, 401)
        xyz = np.zeros((np.size(mesh_x), 3)) #(160801, 3)
        xyz[:, 0] = np.reshape(mesh_x, -1)
        xyz[:, 1] = np.reshape(mesh_y, -1)
        xyz[:, 2] = np.reshape(z_norm, -1)
        print('xyz')
        return xyz, z_norm

    def draw_cloud(self):
        count, dim = self.array.shape
        try: 
            if dim != 3:
                raise ValueError(f'Expected 3 dimensions but got {dim}')
            
            # Visualize point cloud from array
            print(f'Displaying 3D data for {count:,} data points')       
            
            o3d.visualization.draw([self.cloud])#with control panels         

            #visualize a list of o3d.geometry.Geometry objects. Similar results
            o3d.visualization.draw_geometries([self.cloud], mesh_show_wireframe=True)

            depth_reproj = self.cloud.project_to_depth_image(width=640, height=480, intrinsic=None, depth_scale=5000.0, depth_max=10.0)
            

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(np.asarray(self.array.to_legacy()))
            axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
            plt.show()
            
        except Exception as e:
            print(f'Failed to draw point cloud: {e}')
        
    def draw_voxels(self, array):
        try: 
            N = 1000
            pcd = self.create_pcd(array)
            pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
            pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
            o3d.visualization.draw([voxel_grid])

        except Exception as e:
            print(f'Failed to draw voxel grid: {e}')

def test_RGBD():
        #filepath = "data/pointcloud.ply"
    filepath = "sampledata\depth_testresultmono.jpg"
    img = read_image(filepath, rgb=False) #HW format uint8
    o3dimg = o3d.geometry.Image(img) #1 channel 
    o3d.visualization.draw_geometries([o3dimg])

    color_filepath="sampledata\depth_test.jpg"
    color_raw = o3d.io.read_image(color_filepath)
    o3d.visualization.draw_geometries([color_raw])
    depth_raw = o3d.io.read_image(filepath)
    o3d.visualization.draw_geometries([depth_raw])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    rgbd_np=np.asanyarray(rgbd_image)

    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    #PinholeCameraIntrinsicParameters.PrimeSenseDefault as default camera parameter. 
    #It has image resolution 640x480, focal length (fx, fy) = (525.0, 525.0), and optical center (cx, cy) = (319.5, 239.5). 
    #An identity matrix is used as the default extrinsic parameter.
    
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    xyz, z_norm=MyPointCloud.gen_simdata()
    pcd=MyPointCloud.create_pcd(xyz)
    o3d.visualization.draw_geometries([pcd])
    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8)) #1 channel 
    o3d.io.write_image("data/test.png", img)
    o3d.visualization.draw_geometries([img])

    filepath = "sampledata\depth_testresultmono.jpg"
    pp = MyPointCloud(file=filepath)
    pp.draw_cloud()

    pp.draw_voxels(pp.array)