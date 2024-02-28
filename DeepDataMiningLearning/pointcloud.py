import numpy as np
import open3d as o3d
from tqdm import tqdm


class PointCloud(object):
    def __init__(self, array):
        self.array = self.__preprocess(array)
        self.cloud = self.__create_pcd(self.array)
    
    @staticmethod
    def __preprocess(array) -> dict:
        rgb = []
        x, y = array.shape
        avg = lambda d: sum(d) / len(d)
        scale_factor = avg(array.shape)/array.max()
        for row in tqdm(range(y)):
            for col in range(x):
                mod_depth = array[row][col]*scale_factor
                xyz = np.array([row, col, mod_depth])
                rgb.append(xyz)

        rgb = np.array(rgb)
        print(f'Converted raw array of shape {array.shape} to RGB array of shape {rgb.shape}')
        return rgb 
    
    @staticmethod
    def __create_pcd(array):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
        return pcd
        
    def draw_cloud(self):
        count, dim = self.array.shape
        try: 
            if dim != 3:
                raise ValueError(f'Expected 3 dimensions but got {dim}')
            
            # Visualize point cloud from array
            print(f'Displaying 3D data for {count:,} data points')                
            o3d.visualization.draw_geometries([self.cloud])
            
        except Exception as e:
            print(f'Failed to draw point cloud: {e}')
        
    def draw_voxels(self, array):
        try: 
            N = 1000
            pcd = self.__create_pcd(array)
            pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
            pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
            o3d.visualization.draw([voxel_grid])

        except Exception as e:
            print(f'Failed to draw voxel grid: {e}')
    