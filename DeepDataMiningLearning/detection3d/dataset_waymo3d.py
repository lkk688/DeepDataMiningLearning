import os
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

KEY_SEG = "key.segment_context_name"
KEY_TS  = "key.frame_timestamp_micros"
LASER_NAME = "key.laser_name"

RI_VALS1 = "[LiDARComponent].range_image_return1.values"
RI_SHAPE1 = "[LiDARComponent].range_image_return1.shape"

BOX_X = "[LiDARBoxComponent].box.center.x"
BOX_Y = "[LiDARBoxComponent].box.center.y"
BOX_Z = "[LiDARBoxComponent].box.center.z"
BOX_L = "[LiDARBoxComponent].box.size.x"
BOX_W = "[LiDARBoxComponent].box.size.y"
BOX_H = "[LiDARBoxComponent].box.size.z"
BOX_HEADING = "[LiDARBoxComponent].box.heading"
BOX_TYPE = "[LiDARBoxComponent].type"


def _read_list_column(col):
    """Safely convert PyArrow or numpy list to numpy array."""
    if hasattr(col, "values"):
        return np.array([x.as_py() for x in col.values])
    if isinstance(col, (list, np.ndarray)):
        out = []
        for x in col:
            if hasattr(x, "as_py"):
                out.append(x.as_py())
            else:
                out.append(x)
        return np.array(out)
    if hasattr(col, "as_py"):
        return np.array(col.as_py())
    raise TypeError(f"Unsupported column type: {type(col)}")


def _decode_range_image(row):
    vals = _read_list_column(row[RI_VALS1])
    shape = _read_list_column(row[RI_SHAPE1])
    if len(shape) < 3:
        raise ValueError(f"Invalid range image shape: {shape}")
    H, W, C = map(int, shape[:3])
    arr = np.array(vals, dtype=np.float32).reshape(H, W, C)
    return arr


def _spherical_to_cartesian(range_img, inclinations, extrinsic):
    """Convert range image to xyz using beam inclinations and extrinsic."""
    H, W = range_img.shape
    azimuth = np.linspace(-np.pi, np.pi, W, endpoint=False)
    incl = inclinations.reshape(H, 1)
    ranges = range_img

    x = ranges * np.cos(incl) * np.cos(azimuth)
    y = ranges * np.cos(incl) * np.sin(azimuth)
    z = ranges * np.sin(incl)

    pts = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    pts_world = pts @ extrinsic.T
    return pts_world[:, :3]


def _get_calibration_fields(pf: pq.ParquetFile):
    """Return names for beam_inclination and extrinsic if exist; otherwise mark as minmax/identity."""
    names = set(pf.schema.names)
    # beam
    if any("beam_inclination.values" in n for n in names):
        beam_field = [n for n in names if "beam_inclination.values" in n][0]
        beam_mode = "values"
    elif any("beam_inclination.min" in n for n in names) and any("beam_inclination.max" in n for n in names):
        beam_field = None
        beam_mode = "minmax"
    else:
        raise KeyError(f"Cannot find beam_inclination fields in {names}")

    # extrinsic
    extr_field = next((n for n in names if "extrinsic" in n and "transform" in n), None)
    extr_mode = "matrix" if extr_field else "identity"
    return {"beam_mode": beam_mode, "beam_field": beam_field,
            "extr_mode": extr_mode, "extr_field": extr_field}

class Waymo3DDataset(Dataset):
    """Waymo v2.1 → 3D Dataset (range image → point cloud)."""

    def __init__(self, root_dir, split="training", max_frames=None):
        self.root = root_dir
        self.split = split
        self.lidar_dir = os.path.join(root_dir, split, "lidar")
        self.box_dir   = os.path.join(root_dir, split, "lidar_box")
        self.calib_dir = os.path.join(root_dir, split, "lidar_calibration")

        if not os.path.isdir(self.lidar_dir):
            raise FileNotFoundError(self.lidar_dir)
        if not os.path.isdir(self.calib_dir):
            raise FileNotFoundError(self.calib_dir)
        if not os.path.isdir(self.box_dir):
            raise FileNotFoundError(self.box_dir)

        # shard matching
        files = [f for f in os.listdir(self.lidar_dir) if f.endswith(".parquet")]
        valid = [f for f in files if os.path.exists(os.path.join(self.box_dir, f))]
        if not valid:
            raise RuntimeError("No matching lidar/lidar_box shards found")

        self.frame_index = []
        total = 0
        for fname in valid:
            pf = pq.ParquetFile(os.path.join(self.lidar_dir, fname))
            ts = pf.read_row_group(0, columns=[KEY_TS])[KEY_TS].to_numpy()
            seg = pf.read_row_group(0, columns=[KEY_SEG])[KEY_SEG].to_numpy()
            for i, (s, t) in enumerate(zip(seg, ts)):
                self.frame_index.append((fname, i, int(t), s))
                total += 1
                if max_frames and total >= max_frames:
                    break
            if max_frames and total >= max_frames:
                break

        print(f"✅ Waymo3DDataset initialized with {len(self.frame_index)} frames using lidar_calibration/")

    def __len__(self): return len(self.frame_index)

    def __getitem__(self, idx):
        """Load one Waymo LiDAR frame, convert range image to point cloud, and load 3D boxes."""
        fname, row_idx, ts, seg = self.frame_index[idx]

        # --------------------------------------------------
        # 1️⃣ 读取 range image 数据
        # --------------------------------------------------
        lidar_path = os.path.join(self.lidar_dir, fname)
        pf = pq.ParquetFile(lidar_path)
        df = pf.read_row_group(0).to_pandas()
        row = df.iloc[row_idx]
        laser_id = int(row[LASER_NAME])

        # 解码 range image
        ri = _decode_range_image(row)
        if ri is None:
            raise RuntimeError("Range image decode failed.")
        range_img = ri[..., 0]
        intensity = ri[..., 1] if ri.shape[-1] > 1 else np.zeros_like(range_img)

        # --------------------------------------------------
        # 2️⃣ 读取 calibration（beam inclinations + extrinsic）
        # --------------------------------------------------
        calib_path = os.path.join(self.calib_dir, fname)
        cf = pq.ParquetFile(calib_path)
        fields = _get_calibration_fields(cf)
        cdf = cf.read().to_pandas()
        crow = cdf[(cdf[KEY_SEG] == seg) & (cdf["key.laser_name"] == laser_id)]
        if len(crow) == 0:
            raise RuntimeError(f"No calibration for {seg}, laser={laser_id}")
        crow = crow.iloc[0]

        # ---- beam inclinations ----
        if fields["beam_mode"] == "values":
            inc_vals = _read_list_column(crow[fields["beam_field"]])
            inclinations = np.array(inc_vals, dtype=np.float32)
        else:
            # 仅 min/max 可用时：根据 range image 高度生成线性倾角
            minv = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
            maxv = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
            num_beams = ri.shape[0]
            inclinations = np.linspace(minv, maxv, num_beams, dtype=np.float32)

        # ---- extrinsic ----
        if fields["extr_mode"] == "matrix":
            extr_vals = _read_list_column(crow[fields["extr_field"]])
            extrinsic = np.array(extr_vals, dtype=np.float32).reshape(4, 4)
        else:
            extrinsic = np.eye(4, dtype=np.float32)

        # --------------------------------------------------
        # 3️⃣ range image → 点云 (spherical → cartesian)
        # --------------------------------------------------
        xyz = _spherical_to_cartesian(range_img, inclinations, extrinsic)
        inten = intensity.reshape(-1, 1)
        lidar = torch.tensor(np.concatenate([xyz, inten], axis=1), dtype=torch.float32)

        # --------------------------------------------------
        # 4️⃣ 加载 3D box 标注
        # --------------------------------------------------
        box_path = os.path.join(self.box_dir, fname)
        bpf = pq.ParquetFile(box_path)
        bcols = [BOX_X, BOX_Y, BOX_Z, BOX_L, BOX_W, BOX_H, BOX_HEADING, BOX_TYPE, KEY_SEG, KEY_TS]
        bdf = bpf.read_row_group(0, columns=bcols).to_pandas()
        boxes = bdf[(bdf[KEY_SEG] == seg) & (bdf[KEY_TS] == ts)]

        if len(boxes) == 0:
            boxes3d = torch.zeros((0, 7), dtype=torch.float32)
            labels  = torch.zeros((0,), dtype=torch.int64)
        else:
            arr = boxes[[BOX_X, BOX_Y, BOX_Z, BOX_L, BOX_W, BOX_H, BOX_HEADING]].to_numpy()
            boxes3d = torch.tensor(arr, dtype=torch.float32)
            labels  = torch.tensor(boxes[BOX_TYPE].to_numpy(), dtype=torch.int64)

        # --------------------------------------------------
        # 5️⃣ 返回
        # --------------------------------------------------
        target = {
            "boxes_3d": boxes3d,
            "labels": labels,
            "segment": seg,
            "timestamp": ts,
            "laser_id": laser_id,
        }
        return lidar, target

def test_parquet():
    calib_path = "/mnt/e/Shared/Dataset/waymodata/training/lidar_calibration/9758342966297863572_875_230_895_230.parquet"
    pf = pq.ParquetFile(calib_path)
    print(pf.schema)

import open3d as o3d
import numpy as np

def visualize_open3d(lidar: torch.Tensor, boxes3d: torch.Tensor | None = None,
                     point_size: float = 1.0, color_by_intensity: bool = True):
    """
    Visualize LiDAR point cloud + 3D boxes interactively using Open3D.

    Args:
        lidar: torch.Tensor [N,4] -> (x,y,z,intensity)
        boxes3d: torch.Tensor [M,7] -> (x,y,z,dx,dy,dz,yaw)
        point_size: size of points for rendering
        color_by_intensity: if True, map intensity to gray scale
    """
    if not isinstance(lidar, np.ndarray):
        lidar = lidar.cpu().numpy()
    if boxes3d is not None and not isinstance(boxes3d, np.ndarray):
        boxes3d = boxes3d.cpu().numpy()

    # ----- 点云 -----
    pts = lidar[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if lidar.shape[1] > 3 and color_by_intensity:
        inten = lidar[:, 3]
        inten = (inten - inten.min()) / (inten.ptp() + 1e-6)
        colors = np.stack([inten, inten, inten], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geoms = [pcd]

    # ----- 3D boxes -----
    if boxes3d is not None and len(boxes3d) > 0:
        for b in boxes3d:
            x, y, z, dx, dy, dz, yaw = b
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            obb = o3d.geometry.OrientedBoundingBox(center=[x, y, z],
                                                   R=R,
                                                   extent=[dx, dy, dz])
            obb.color = (0, 1, 0)
            geoms.append(obb)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Waymo LiDAR + 3D Boxes", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

def download_waymo_folder(LOCAL_DIR = "/data/Datasets/waymodata/", SPLIT="training"):
    import os
    import subprocess

    # Configuration
    BUCKET_PREFIX = "gs://waymo_open_dataset_v_2_0_1/training/"
    
    # Ensure the local directory exists
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # List all subdirectories in the training bucket
    # Using `gsutil ls` with a trailing '/' lists only folders.
    command = f"gsutil ls {BUCKET_PREFIX}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        folders_to_download = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error listing bucket contents: {e.stderr}")
        exit(1)

    print(f"Found {len(folders_to_download)} folders in the bucket.")

    for remote_folder in folders_to_download:
        folder_name = os.path.basename(remote_folder.strip('/'))
        local_folder_path = os.path.join(LOCAL_DIR, SPLIT, folder_name)

        # Check if the local folder already exists
        if os.path.exists(local_folder_path) or folder_name==SPLIT:
            print(f"Skipping existing folder: {local_folder_path}")
        else:
            print(f"Downloading new folder: {folder_name}...")
            os.makedirs(local_folder_path, exist_ok=True)
            # Fix: Copy contents of remote folder to local folder, not the folder itself
            # Add trailing /* to copy all files inside the remote folder
            remote_folder_contents = remote_folder.rstrip('/') + '/*'
            download_command = f"gsutil -m cp -r {remote_folder_contents} {local_folder_path}/"
            try:
                subprocess.run(download_command, shell=True, check=True)
                print(f"Successfully downloaded {folder_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {folder_name}: {e.stderr}")

    print("Download script finished.")

def main():
    #path="/mnt/e/Shared/Dataset/waymodata/"
    ds = Waymo3DDataset("/data/Datasets/waymodata/", split="training", max_frames=3)
    lidar, target = ds[0]
    print("points:", lidar.shape)
    print("boxes:", target["boxes_3d"].shape)

    visualize_open3d(lidar, target["boxes_3d"])

if __name__ == "__main__":
    download_waymo_folder()
    main()
    #test_parquet()