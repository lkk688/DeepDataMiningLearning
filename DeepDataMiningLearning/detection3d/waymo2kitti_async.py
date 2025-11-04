r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""


import os
import cv2
#import tensorflow.compat.v1 as tf
import math
import itertools
import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join
from os import path as osp
from multiprocessing import Pool #https://docs.python.org/3/library/multiprocessing.html
import asyncio #https://asyncio.readthedocs.io/en/latest/hello_world.html
import time
import concurrent.futures

#tf.enable_eager_execution()
try:
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

# from waymo_open_dataset.utils import range_image_utils, transform_utils
# from waymo_open_dataset.utils.frame_utils import \
#     parse_range_image_and_camera_projection

class Waymo2KITTIAsync(object):
    """Waymo to KITTI converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        workers (str): Number of workers for the parallel process.
        test_mode (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 alltfrecordfiles,
                 #load_dir,
                 save_dir,
                 workers=4,
                 startingindex=0,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        #self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST','SIGN']
        print("selected_waymo_classes:", self.selected_waymo_classes)

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.startingindex = startingindex
        print("startingindex:", startingindex)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        #self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.test_mode = test_mode
        self.totalimage_count=0

        self.tfrecord_pathnames = alltfrecordfiles
        self.totalfilenum =len(self.tfrecord_pathnames)
#         sorted(
#             glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()
    
    def convertcoroutine(self):
        _start = time.time()
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.mycoroutinetaskprocess())#(self.concurrenttaskthread())
        loop.run_until_complete(future)
        print(f"Execution time: { time.time() - _start }")
        loop.close()
    
    async def mycoroutinetask(self):
        tasks = []
        print('Start converting ...')
        for file_idx in range(self.startingindex, self.totalfilenum):
            print("Current: fileindex:", file_idx)
            task = asyncio.ensure_future(self.convert_one(file_idx))
            #self.convert_one(file_idx)
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def mycoroutinetaskprocess(self):#https://github.com/jersobh/python-parallelism-examples/blob/master/async_concurrent_futures.py
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.convert_one, file_idx) for file_idx in range(self.startingindex, self.totalfilenum) }
            for future in concurrent.futures.as_completed(futures):
                print("finished one future")
    
    def concurrenttaskthread(self):#https://github.com/jersobh/python-parallelism-examples/blob/master/async_concurrent_futures.py
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.convert_one_async, file_idx) for file_idx in range(self.startingindex, self.totalfilenum) }
            for future in concurrent.futures.as_completed(futures):
                print("finished one future")

    def convert(self):
        """Convert action."""
        import mmcv
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_singlethread(self):
        print('Start converting ...')
        for file_idx in range(self.startingindex, self.totalfilenum):
            print("Current: fileindex:", file_idx)
            self.convert_one(file_idx)
#         for file_idx in range(len(self.tfrecord_pathnames)):
#             print("Current: fileindex:", file_idx)
#             self.convert_one(file_idx)
    
    def convert_multithread(self):
        print('Start multithread converting ...')
        chunksize =1
        process_num=self.workers
        skip_first=False
        tasks = range(len((self.tfrecord_pathnames)))
        task_num = len(tasks)
        pool = Pool(process_num)
        results = []
        gen = pool.imap(self.convert_one, tasks, chunksize)
        for result in gen:
            results.append(result)
        pool.close()
        pool.join()
        
    def convert_one_async(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        print("Convert: fileindex:", file_idx)
        print("Current path:", pathname)
        c_start = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        for frame_idx, data in enumerate(dataset):
            loop.run_until_complete(self.savetofiles(data, file_idx, frame_idx))
            #loop.run_until_complete(self.do_stuff(file_idx, frame_idx))
        loop.close()

        # dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        # loop = asyncio.get_event_loop()
        # for frame_idx, data in enumerate(dataset):
        #     loop.run_until_complete(savetofiles(data, file_idx, frame_idx))
        #     break
        # loop.close()
            #await savetofiles(data, file_idx, frame_idx)
#             loop = asyncio.get_event_loop()
#             loop.run_until_complete(savetofiles(data, file_idx, frame_idx))
#             loop.close()
            
        print(f"convert one Execution time: { time.time() - c_start }")
        
    async def do_stuff(self,file_idx, frame_idx):
        import random
        await asyncio.sleep(random.uniform(0.1, 0.5)) # NOTE if we hadn't called
            # asyncio.set_event_loop() earlier, we would have to pass an event
            # loop to this function explicitly.
        print("in do stuff: ",file_idx, frame_idx)

    async def savetofiles(self,data, file_idx, frame_idx):
        print("savetofiles:",file_idx,frame_idx )
        c_start = time.time()
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        self.save_image(frame, file_idx, frame_idx)
        self.save_calib(frame, file_idx, frame_idx)
        self.save_lidar(frame, file_idx, frame_idx)
        self.save_pose(frame, file_idx, frame_idx)

        if not self.test_mode:
            self.save_label(frame, file_idx, frame_idx)
        print(f"savetofiles fileid:{file_idx} frameid:{frame_idx} Execution time: { time.time() - c_start }")

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        print("Convert: fileindex:", file_idx)
        print("Current path:", pathname)
        c_start = time.time()
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            self.save_image(frame, file_idx, frame_idx)
            self.save_calib(frame, file_idx, frame_idx)
            self.save_lidar(frame, file_idx, frame_idx)
            self.save_pose(frame, file_idx, frame_idx)

            if not self.test_mode:
                self.save_label(frame, file_idx, frame_idx)
        print(f"convert one Execution time: { time.time() - c_start }")

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in png format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            # img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
            #     f'{self.prefix}{str(file_idx).zfill(3)}' + \
            #     f'{str(frame_idx).zfill(3)}.png'

            foldername = f'{self.image_save_dir}{str(img.name - 1)}/'
            img_path = foldername + \
                f'{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.png'
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            imag = tf.image.decode_jpeg(img.image).numpy()
            self.totalimage_count=self.totalimage_count+1
            cv2.imwrite(str(img_path), cv2.cvtColor(imag, cv2.COLOR_RGB2BGR))
        # img = mmcv.imfrombytes(img.image)
        # mmcv.imwrite(img, img_path)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters, Camera frame to vehicle frame.
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # range_images, camera_projections, range_image_top_pose = \
        #     parse_range_image_and_camera_projection(frame)

        #convert_range_image_to_point_cloud 
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            keep_polar_features=True)
        
        # 3d points in vehicle frame. 
        points_all = np.concatenate(points, axis=0) #combines 5 lidar data together
        # declare new index list 
        i = [3,4,5,1] 
        # create output 
        pointsxyzintensity_output = points_all[:,i] 

        pc_path = f'{self.point_cloud_save_dir}/' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        pointsxyzintensity_output.astype(np.float32).tofile(pc_path)

        # # First return
        # points_0, cp_points_0, intensity_0, elongation_0 = \
        #     self.convert_range_image_to_point_cloud(
        #         frame,
        #         range_images,
        #         camera_projections,
        #         range_image_top_pose,
        #         ri_index=0
        #     )
        # points_0 = np.concatenate(points_0, axis=0)
        # intensity_0 = np.concatenate(intensity_0, axis=0)
        # elongation_0 = np.concatenate(elongation_0, axis=0)

        # # Second return
        # points_1, cp_points_1, intensity_1, elongation_1 = \
        #     self.convert_range_image_to_point_cloud(
        #         frame,
        #         range_images,
        #         camera_projections,
        #         range_image_top_pose,
        #         ri_index=1
        #     )
        # points_1 = np.concatenate(points_1, axis=0)
        # intensity_1 = np.concatenate(intensity_1, axis=0)
        # elongation_1 = np.concatenate(elongation_1, axis=0)

        # points = np.concatenate([points_0, points_1], axis=0)
        # intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        # elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        # # Lidar elongation refers to the elongation of the pulse beyond its nominal width
        # point_cloud = np.column_stack(
        #     (points, intensity, elongation, timestamp))

        # pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
        #     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        # point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')#Opens a file for writing and reading, w+ will "overwrite the existing file if the file exists"
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels: #frame.camera_labels: #frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1
        #print("id_to_bbox:",id_to_bbox)
        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                # print("Lidar:", lidar)
                # print("boundingbox:", id_to_bbox.get(id))
                # print("boundingbox id lidar:", id_to_bbox.get(id + lidar))
                if id + lidar in id_to_bbox:
                    #print("id + lidar:",id + lidar)
                    bounding_box = id_to_bbox.get(id + lidar)
                    #print("bounding_box:", bounding_box)
                    name = str(id_to_name.get(id + lidar))
                    #print("name:",name)
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')#Opens a file for appending new information to it.
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            #mmcv.mkdir_or_exist(d)
            self.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                #mmcv.mkdir_or_exist(f'{d}{str(i)}')
                self.mkdir_or_exist(f'{d}{str(i)}')

    def mkdir_or_exist(self,foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername)

    
    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == "__main__":
    #folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031","validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]#["training_0001"]# ["training_0000", "training_0001"]
    # folders = ["training_0004"]
    # folder_name="4c_train0004"
    folders = ["training_0005", "training_0006","training_0007","training_0008"]
    folder_name="4c_train5678"
    root_path="/data/cmpe249-f20/Waymo"
    out_dir="/data/cmpe249-f20/WaymoKittitMulti"
    data_files = [path for x in folders for path in glob(os.path.join(root_path, x, "*.tfrecord"))]
    print("totoal number of files:", len(data_files))

    workers=56 #48 #16
    c_start = time.time()
    print(c_start)
    save_dir = osp.join(out_dir, folder_name, 'training')
    converter = Waymo2KITTIAsync(
            data_files,
            save_dir,
            workers=workers,
            startingindex=0,
            test_mode=False)
    converter.concurrenttaskthread()#convert_multithread()#convertcoroutine()#concurrenttaskthread()#.convert()
    
    # for i, split in enumerate(folders):
    #     #load_dir = osp.join(root_path, 'waymo_format', split)
    #     load_dir = osp.join(root_path, split)
    #     if split == 'validation':
    #         save_dir = osp.join(out_dir, 'validation')
    #     else:
    #         save_dir = osp.join(out_dir, 'training', split)
    #     converter = Waymo2KITTIAsync(
    #         data_files,
    #         save_dir,
    #         workers=workers,
    #         test_mode=(split == 'test'))
    #     converter.concurrenttaskthread()#convert_multithread()#convertcoroutine()#concurrenttaskthread()#.convert()
    
    print(f"Finished, Execution time: { time.time() - c_start }")