import numpy as np
import pyarrow.parquet as pq

# === 1️⃣ Load one frame (choose one file + segment + timestamp) ===
root = "/data/Datasets/waymodata/training"
fname = "6148393791213790916_4960_000_4980_000.parquet"   # example filename
seg   = "6148393791213790916_4960_000_4980_000"                    # segment id
ts    = 1515524990924880                                      # frame timestamp (μs)
laser_id = 1  # typically 1 = TOP LiDAR

# --- Read range image ---
pf = pq.ParquetFile(f"{root}/lidar/{fname}")
df = pf.read_row_group(0).to_pandas()
row = df[(df["key.segment_context_name"] == seg) &
          (df["key.frame_timestamp_micros"] == ts) &
          (df["key.laser_name"] == laser_id)].iloc[0]

vals = row["[LiDARComponent].range_image_return1.values"].as_py()
shape = row["[LiDARComponent].range_image_return1.shape"].as_py()
ri = np.array(vals, np.float32).reshape(shape, order="C")     # [H,W,4]

rng = np.clip(np.nan_to_num(ri[..., 0], nan=0.0), 0.0, 300.0)
H, W = rng.shape
print(f"[INFO] Range image shape: {ri.shape}")

# === 2️⃣ Decode LiDAR calibration ===
pf_cal = pq.ParquetFile(f"{root}/lidar_calibration/{fname}")
df_cal = pf_cal.read_row_group(0).to_pandas()
crow = df_cal[(df_cal["key.segment_context_name"] == seg) &
              (df_cal["key.laser_name"] == laser_id)].iloc[0]

inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
inclinations = np.linspace(inc_min, inc_max, H, dtype=np.float32)
if np.max(np.abs(inclinations)) > np.pi:
    inclinations = np.deg2rad(inclinations)

# Extrinsic: LiDAR → Vehicle (row-major)
extr = np.array(
    crow[[c for c in crow.index if "extrinsic" in c or str(c).endswith("item")][0]].as_py(),
    np.float32
).reshape(4, 4, order="C")

print("[INFO] LiDAR→Vehicle extrinsic:\n", extr)

# === 3️⃣ Range image → LiDAR Cartesian ===
incl = inclinations[::-1].reshape(H, 1)
az = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)

cos_i, sin_i = np.cos(incl), np.sin(incl)
cos_a, sin_a = np.cos(az), np.sin(az)

Xl = rng * cos_i * cos_a
Yl = rng * cos_i * sin_a
Zl = rng * sin_i

pts_l = np.stack([Xl, Yl, Zl, np.ones_like(Zl)], axis=-1).reshape(-1, 4)

# === 4️⃣ LiDAR → Vehicle ===
pts_vehicle = (pts_l @ extr.T)[:, :3]

print(f"[DEBUG] pts_vehicle range X[{pts_vehicle[:,0].min():.1f},{pts_vehicle[:,0].max():.1f}] "
      f"Y[{pts_vehicle[:,1].min():.1f},{pts_vehicle[:,1].max():.1f}] Z[{pts_vehicle[:,2].min():.1f},{pts_vehicle[:,2].max():.1f}]")

# === 5️⃣ Vehicle → World ===
pf_pose = pq.ParquetFile(f"{root}/vehicle_pose/{fname}")
df_pose = pf_pose.read_row_group(0).to_pandas()
prow = df_pose[(df_pose["key.segment_context_name"] == seg) &
               (df_pose["key.frame_timestamp_micros"] == ts)].iloc[0]

T_wv = np.array(
    prow["[VehiclePoseComponent].world_from_vehicle.transform"].as_py(),
    np.float32
).reshape(4, 4, order="C")

pts_vh = np.concatenate([pts_vehicle, np.ones((pts_vehicle.shape[0], 1), np.float32)], axis=1)
pts_world = (pts_vh @ T_wv.T)[:, :3]

print("[INFO] Vehicle→World translation:", T_wv[:3, 3])
print(f"[DEBUG] pts_world range X[{pts_world[:,0].min():.1f},{pts_world[:,0].max():.1f}] "
      f"Y[{pts_world[:,1].min():.1f},{pts_world[:,1].max():.1f}] Z[{pts_world[:,2].min():.1f},{pts_world[:,2].max():.1f}]")

# === 6️⃣ Load 3D boxes (already in Vehicle frame) ===
pf_box = pq.ParquetFile(f"{root}/lidar_box/{fname}")
df_box = pf_box.read_row_group(0).to_pandas()
rows = df_box[(df_box["key.segment_context_name"] == seg) &
              (df_box["key.frame_timestamp_micros"] == ts)]
arr = rows[[
    "[LiDARBoxComponent].box.center.x",
    "[LiDARBoxComponent].box.center.y",
    "[LiDARBoxComponent].box.center.z",
    "[LiDARBoxComponent].box.size.x",
    "[LiDARBoxComponent].box.size.y",
    "[LiDARBoxComponent].box.size.z",
    "[LiDARBoxComponent].box.heading",
]].to_numpy().astype(np.float32)

arr[:, 6] = -arr[:, 6]   # Waymo heading is clockwise → negate
boxes3d_vehicle = arr

print(f"[INFO] Loaded {len(boxes3d_vehicle)} boxes in vehicle frame.")