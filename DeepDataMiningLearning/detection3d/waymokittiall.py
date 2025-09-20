#based on drawkittilidar.py
import numpy as np
import os
import cv2
import sys
import argparse
import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
import mayavi.mlab as mlab

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#Current folder
ROOT_DIR = os.path.dirname(BASE_DIR)#Project root folder
sys.path.append(ROOT_DIR)

from CalibrationUtils import WaymoCalibration, KittiCalibration, rotx, roty, rotz
from mydetector3d.tools.visual_utils.mayavivisualize_utils import visualize_pts, draw_lidar, draw_gt_boxes3d, draw_scenes #, pltlidar_with3dbox

class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_diffculty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
            bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


def filter_lidarpoints(pc_velo, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    #Filter Lidar Points
    #point_cloud_range=[0, -15, -5, 90, 15, 4]#[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    mask = (pc_velo[:, 0] >= point_cloud_range[0]) & (pc_velo[:, 0] <= point_cloud_range[3]) \
           & (pc_velo[:, 1] >= point_cloud_range[1]) & (pc_velo[:, 1] <= point_cloud_range[4]) \
           & (pc_velo[:, 2] >= point_cloud_range[2]) & (pc_velo[:, 2] <= point_cloud_range[5]) \
           & (pc_velo[:, 3] <= 1) 
    filteredpoints=pc_velo[mask] #(43376, 4)
    print(filteredpoints.shape)
    return filteredpoints


def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def read_label(label_filename):
    if os.path.exists(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects
    else:
        return []

def read_multi_label(label_files):
    objectlabels=[]
    for label_file in label_files:
        object3dlabel=read_label(label_file)
        objectlabels.append(object3dlabel)
    return objectlabels
import matplotlib.pyplot as plt
import matplotlib.patches as patches

INSTANCE_Color = {
    'Car':'red', 'Pedestrian':'green', 'Sign': 'yellow', 'Cyclist':'purple'
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'

INSTANCE3D_ColorCV2 = {
    'Car':(0, 255, 0), 'Pedestrian':(255, 255, 0), 'Sign': (0, 255, 255), 'Cyclist':(127, 127, 64)
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'


waymocameraorder={
        0:1, 1:0, 2:2, 3:3, 4:4
    }#Front, front_left, side_left, front_right, side_right
cameraname_map={0:"FRONT", 1:"FRONT_LEFT", 2:"FRONT_RIGHT", 3:"SIDE_LEFT", 4:"SIDE_RIGHT"}

def pltlidar_with3dbox(pc_velo, object3dlabels, calib, point_cloud_range):
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(pc_velo, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=True, drawregion=True, point_cloud_range=point_cloud_range)
    #visualize_pts(pc_velo, fig=fig, show_intensity=True)

    #only draw camera 0's 3D label
    ref_cameraid=0 #3D labels are annotated in camera 0 frame
    color = (0, 1, 0)
    for obj in object3dlabels:
        if obj.type == "DontCare":
            continue
        #print(obj.type)
        # Draw 3d bounding box
        box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid) #(n,8,3)
        #print("box3d_pts_3d_velo:", box3d_pts_3d_velo)
        #draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
        if obj.type in INSTANCE3D_Color.keys():
            colorlabel=INSTANCE3D_Color[obj.type]
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=colorlabel, label=obj.type) #(n,8,3)
        else:
            print("Object not in Kitti:", obj.type)

    mlab.show()

def plt_multiimages(images, objectlabels, datasetname, order=1):
    plt.figure(order, figsize=(16, 9))
    camera_count = len(images)
    for count in range(camera_count):#each frame has 5 images
        if datasetname.lower()=='waymokitti':
            index=waymocameraorder[count]
            pltshow_image_with_boxes(index, images[index], objectlabels[index], [3, 3, count+1])
        elif datasetname.lower()=='kitti':
            index=count
            pltshow_image_with_boxes(index, images[index], objectlabels[index], [1, 2, count+1])
        

def pltshow_image_with_boxes(cameraid, img, objects, layout, cmap=None):
    ax = plt.subplot(*layout)
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    plt.imshow(img1, cmap=cmap)
    plt.title(cameraname_map[cameraid])
    
    if not objects or len(objects)==0: #no objects
        return
    for obj in objects:
        if obj.type == "DontCare":
            continue
        box=obj.box2d
        objectclass=obj.type
        if objectclass in INSTANCE_Color.keys():
            colorlabel=INSTANCE_Color[objectclass]
            [xmin, ymin, xmax, ymax]=box
            width=xmax-xmin #box.length
            height=ymax-ymin #box.width
            if (height>0 and width>0):
                #print(box)
    #             xmin=label.box.center_x - 0.5 * label.box.length
    #             ymin=label.box.center_y - 0.5 * label.box.width
                # Draw the object bounding box.
                ax.add_patch(patches.Rectangle(
                    xy=(xmin,ymin),
                    width=width, #label.box.length,
                    height=height, #label.box.width,
                    linewidth=1,
                    edgecolor=colorlabel,
                    facecolor='none'))
                ax.text(xmin, ymin, objectclass, color=colorlabel, fontsize=8)
        else:
            print("Object not in kitti:", objectclass)
    # Show the camera image.
    plt.grid(False)
    plt.axis('on')


def plt3dbox_images(images,objectlabels,calib, datasetname='kitti'):
    plt.figure(figsize=(16, 9))
    camera_count = len(images)
    for count in range(camera_count):#each frame has 5 images
        if datasetname.lower()=='waymokitti':
            index=waymocameraorder[count]
            img = images[index]
            object3dlabel=objectlabels[index]
            pltshow_image_with_3Dboxes(index, img, object3dlabel,calib, [3, 3, count+1])
        elif datasetname.lower()=='kitti':
            index=count
            img = images[index]
            object3dlabel=objectlabels[index]
            pltshow_image_with_3Dboxes(index, img, object3dlabel,calib, [1, 2, count+1])

def pltshow_image_with_3Dboxes(cameraid, img, objects, calib, layout, cmap=None):
    ax = plt.subplot(*layout)
    """ Show image with 3D bounding boxes """
    img2 = np.copy(img)  # for 3d bbox
    #plt.figure(figsize=(25, 20))
    print("camera id:", cameraid)
    z_front_min = -3 #0.1 #0.1
    if cameraid ==0:
        for obj in objects:
            if obj.type == "DontCare" or (obj is None):
                continue
            box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 8x3
            #print(box3d_pts_3d)
            if np.any(box3d_pts_3d[2, :] < z_front_min): #in Kitti, z axis is to the front, if z<0.1 means objs in back of camera
                continue
            box3d_pts_2d, _ = calib.project_cam3d_to_image(box3d_pts_3d, cameraid) #return (8,2) array in left image coord.
            #print("obj:", box3d_pts_2d)
            if box3d_pts_2d is not None:
                if obj.type in INSTANCE3D_ColorCV2.keys():
                    colorlabel=INSTANCE3D_ColorCV2[obj.type]
                    img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)
                else:
                    print("Object not in kitti:", obj.type)
    else:
        ref_cameraid=0
        for obj in objects:
            if obj.type == "DontCare" or (obj is None):
                continue
            #_, box3d_pts_3d = compute_box_3d(obj, calib.P[camera_index]) #get 3D points in label (in camera 0 coordinate), convert to 8 corner points
            box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 
            # if np.any(box3d_pts_3d[2, :] < z_front_min): #in Kitti, z axis is to the front, if z<0.1 means objs in back of camera
            #     continue
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid) # convert the 3D points to velodyne coordinate
            box3d_pts_3d_cam=calib.project_velo_to_cameraid(box3d_pts_3d_velo,cameraid) # convert from the velodyne coordinate to camera coordinate (cameraid)
            box3d_pts_2d, _=calib.project_cam3d_to_image(box3d_pts_3d_cam,cameraid) # project 3D points in cameraid coordinate to the imageid coordinate (2D 8 points)
            if box3d_pts_2d is not None:
                print(box3d_pts_2d)
                colorlabel=INSTANCE3D_ColorCV2[obj.type]
                img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)

    plt.imshow(img2, cmap=cmap)
    plt.title(cameraname_map[cameraid])
    plt.grid(False)
    plt.axis('on')


def load_image(img_filenames, jpgfile=False):
    imgs=[]
    for img_filename in img_filenames:
        if jpgfile==True:
            img_filename=img_filename.replace('.png', '.jpg')
        img = cv2.imread(img_filename)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)
    return imgs
    #return cv2.imread(img_filename)

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4, filterpoints=False, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    scan = np.fromfile(velo_filename, dtype=dtype) #(254452,)
    scan = scan.reshape((-1, n_vec))
    xpoints=scan[:,0]
    ypoints=scan[:,1]
    zpoints=scan[:,2]
    print(f"Xrange: {min(xpoints)} {max(xpoints)}")
    print(f"Yrange: {min(ypoints)} {max(ypoints)}")
    print(f"Zrange: {min(zpoints)} {max(zpoints)}")
    if filterpoints:
        print(f"Filter point range, x: {point_cloud_range[0]}, {point_cloud_range[3]}; y: {point_cloud_range[1]}, {point_cloud_range[4]}; z: {point_cloud_range[2]}, {point_cloud_range[5]}")
        scan=filter_lidarpoints(scan, point_cloud_range) #point_cloud_range #0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    return scan


def compute_box_3d(obj, dataset='kitti'):
    """ Takes an object3D
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    """
    #x-y-z: front-left-up (waymo) -> x_right-y_down-z_front(kitti)
    # compute rotational matrix around yaw axis (camera coord y pointing to the bottom, thus Yaw axis is rotate y-axis)
    R = roty(obj.ry)

    # 3d bounding box dimensions: x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
    l = obj.l #x
    w = obj.w #z
    h = obj.h #y

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    #print(corners_3d)
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    # if np.any(corners_3d[2, :] < 0.1): #in Kitti, z axis is to the front, if z<0.1 means objs in back of camera
    #     return np.transpose(corners_3d)
    
    return np.transpose(corners_3d)



    
    

def plotlidar_to_image(pts_3d, img, calib, cameraid=0):
    """ Project 3d points to image plane.
    """
    fig = plt.figure(figsize=(16, 9))
    # draw image
    plt.imshow(img)

    pts_2d, pts_depth = calib.project_velo_to_image(pts_3d, cameraid) 
    #step1 project_velo_to_cameraid_rect: transfer velo to cameraid frame, then apply R to camera rect frame 
    #step2 project_cam3d_to_image: project the 3d points to the camera coordinate

    # remove points outside the image
    inds = pts_2d[:, 0] > 0
    inds = np.logical_and(inds, pts_2d[:, 0] < img.shape[1])
    inds = np.logical_and(inds, pts_2d[:, 1] > 0)
    inds = np.logical_and(inds, pts_2d[:, 1] < img.shape[0])
    inds = np.logical_and(inds, pts_depth > 0)
    
    plt.scatter(pts_2d[inds, 0], pts_2d[inds, 1], c=-pts_depth[inds], alpha=0.5, s=1, cmap='viridis')

    # fig.patch.set_visible(False)
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig('data/kitti_cloud_to_img.png', bbox_inches='tight')
    plt.show()

INSTANCE3D_Color = {
    'Car':(0, 1, 0), 'Pedestrian':(0, 1, 1), 'Sign': (1, 1, 0), 'Cyclist':(0.5, 0.5, 0.3)
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'



def datasetinfo(datasetname):
    if datasetname.lower()=='waymokitti':
        camera_index = 0 # front camera of Waymo is image_0
        max_cameracount = 5
    elif datasetname.lower()=='kitti':
        camera_index = 2 # front camera of Kitti is image_2
        max_cameracount = 5
    return camera_index, max_cameracount

def getcalibration(datasetname, calibration_file):
    if datasetname.lower()=='waymokitti':
        calib=WaymoCalibration(calibration_file)
    elif datasetname.lower()=='kitti':
        calib=KittiCalibration(calibration_file)
    return calib

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default='/mnt/f/Dataset/DAIR-C/infrastructure-side-point-cloud-kitti/training', help="root folder"
    )#'./data/waymokittisample'
    parser.add_argument(
        "--index", default="4534", help="file index"
    )
    parser.add_argument(
        "--dataset", default="kitti", help="dataset name" 
    )#waymokitti
    parser.add_argument(
        "--camera_count", default=1, help="Number of cameras used"
    )
    parser.add_argument(
        "--jpgfile", default=True, help="Number of cameras used"
    )
    args = parser.parse_args()

    basedir = args.root_path
    idx = int(args.index)
    camera_count=args.camera_count

    camera_index, max_cameracount = datasetinfo(args.dataset)

    """Load and parse a velodyne binary file."""
    filename="%06d.png" % (idx)
    image_folder='image_'+str(camera_index)
    #image_file = os.path.join(basedir, image_folder, filename)
    image_files = [os.path.join(basedir, "image_"+str(i+camera_index), filename) for i in range(camera_count)] 
    calibration_file = os.path.join(basedir, 'calib', filename.replace('png', 'txt'))
    label_all_file = os.path.join(basedir, 'label_all', filename.replace('png', 'txt')) #'label_0'
    labels_files=[os.path.join(basedir, "label_"+str(i+camera_index), filename.replace('png', 'txt')) for i in range(camera_count)] 
    lidar_filename = os.path.join(basedir, 'velodyne', filename.replace('png', 'bin'))

    #load Lidar points
    dtype=np.float32
    point_cloud_range=[-100, -60, -8, 100, 60, 8]#[0, -15, -5, 90, 15, 4] #0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    pc_velo=load_velo_scan(lidar_filename, dtype=np.float32, n_vec=4, filterpoints=True, point_cloud_range=point_cloud_range)
    ##Each point encodes XYZ + reflectance in Velodyne coordinate: x = forward, y = left, z = up

    #calib=WaymoCalibration(calibration_file)
    calib=getcalibration(args.dataset, calibration_file)

    images=load_image(image_files, jpgfile=args.jpgfile)
    objectlabels=read_multi_label(labels_files)
    plt_multiimages(images, objectlabels, args.dataset)

    plt3dbox_images(images,objectlabels,calib)

    plotlidar_to_image(pc_velo, images[0], calib, cameraid=0)

    if args.dataset.lower()=='waymokitti':
        object3dlabels=read_label(label_all_file)
    elif args.dataset.lower()=='kitti':
        object3dlabels=objectlabels[0]
    pltlidar_with3dbox(pc_velo, object3dlabels, calib, point_cloud_range)
    
    #draw_scenes(pc_velo, gt_boxes=object3dlabels, ref_boxes=None, ref_scores=None, ref_labels=None)#gt_boxes need ndarray

    #V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

    print("end of demo")
    


