# Option A: GT-LiDAR observation to the driver — precise implementation plan

**Goal:** feed the driver GT geometry (LiDAR points rendered from the reconstruction, via the sensorsim
`render_lidar` RPC) so the +occ arm avoids collisions using real geometry instead of the domain-gap-broken
DA3 monocular depth. Enables the clean **GT-occ vs no-occ** closed-loop upper-bound experiment.

**Why the runtime (not the driver):** `LidarRenderRequest` needs `dynamic_objects` (actor poses) + scene_id
+ sensor_pose + the sensorsim stub. Only the **runtime** has all of these (it builds them for RGB rendering
in `sensorsim_service.construct_rgb_render_request`). The driver is a pure policy with no sensorsim channel,
no actor data. So the runtime must render LiDAR and push it to the driver, mirroring the camera path.

## Camera path to mirror (data flow)
`sensorsim_service.render()` (RGB) → `ImageWithMetadata` → runtime submits via
`driver_service.submit_image()` → RPC `submit_image_observation(RolloutCameraImage)` → driver
`EgodriverServicer.submit_image_observation` stores into `Session` → `_prepare_camera_images` →
`PredictionInput.camera_images` → model.

## Steps (exact locations)

### 1. proto — DONE
`src/grpc/alpasim_grpc/v0/egodriver.proto`: added `rpc submit_lidar_observation (RolloutLidar)` + message
`RolloutLidar { session_uuid; LidarPoints{frame_start_us,frame_end_us,point_xyzs_buffer(bytes float32[N*3]),
num_points,logical_id} lidar; }`.

### 2. sensorsim_service.py — add `construct_lidar_render_request` + `render_lidar`
Mirror `construct_rgb_render_request` (lines ~225-316). Reuse the SAME `trajectory_to_pose_pair` +
`dynamic_objects` construction. Build:
```python
from alpasim_grpc.v0.sensorsim_pb2 import LidarRenderRequest, LidarSpec, LidarDeviceType
def construct_lidar_render_request(self, ego_trajectory, traffic_trajectories, trigger, scene_id):
    # start_us/end_us + trajectory_to_pose_pair + dynamic_objects  (copy from RGB version)
    sensor_pose = trajectory_to_pose_pair(ego_trajectory, delta=None)   # LiDAR ~ rig origin
    return LidarRenderRequest(scene_id=scene_id, lidar_config=LidarSpec(lidar_type=LidarDeviceType.PANDAR128),
        frame_start_us=start_us, frame_end_us=end_us, sensor_pose=sensor_pose, dynamic_objects=dynamic_objects)
async def render_lidar(self, ego_trajectory, traffic_trajectories, trigger, scene_id) -> np.ndarray:
    req = self.construct_lidar_render_request(...)
    resp = await profiled_rpc_call("render_lidar","sensorsim", self.stub.render_lidar, req, ...)
    # resp.point_xyzs (repeated float) or resp.point_xyzs_buffer (bytes) -> (N,3) rig frame
    return points_np
```
NOTE: confirm `SensorsimServiceStub` exposes `render_lidar` (sensorsim.proto line 20 declares it).

### 3. event_loop.py — call render_lidar + submit where cameras render+submit
Rendering is via the pluggable `RendererService` + render-event queue (`make_initial_render_event`,
`queue.submit`). Trace where a step's RGB is produced and `driver_service.submit_image` is called (search
`submit_image` across `src/runtime/alpasim_runtime/` — likely in a render-event handler or step callback).
At that SAME site (has ego_trajectory, traffic_trajectories, trigger, scene_id): if lidar enabled, call
`await sensorsim_service.render_lidar(...)` then `await driver_service.submit_lidar(points, trigger, session)`.
Gate behind a `simulation_config.send_lidar` flag (mirror `send_recording_ground_truth`, config.py:323).

### 4. driver_service.py — add `submit_lidar` client (mirror submit_image, lines ~101-124)
```python
async def submit_lidar(self, points_xyz: np.ndarray, trigger, session_info):
    req = RolloutLidar(session_uuid=..., lidar=RolloutLidar.LidarPoints(
        frame_start_us=..., frame_end_us=..., num_points=len(points_xyz),
        point_xyzs_buffer=points_xyz.astype('<f4').tobytes(), logical_id="lidar_top"))
    await self._call("submit_lidar_observation", self.stub.submit_lidar_observation, req)
```

### 5. driver main.py — handler + Session storage + PredictionInput
- `Session`: add `lidar_points: list[tuple[int, np.ndarray]] = []` + `add_lidar(ts, pts)`.
- Servicer: `async def submit_lidar_observation(self, request, context)` → decode
  `np.frombuffer(request.lidar.point_xyzs_buffer, '<f4').reshape(-1,3)` → `session.add_lidar(...)`.
- `PredictionInput`: add `lidar_points: np.ndarray | None`. Populate in the predict builder (line ~777)
  from `session.lidar_points[-1]` (latest).

### 6. OccModel — use lidar instead of DA3 (occ_mode="lidar")
`_obstacle_profile` from lidar points (rig frame: x forward, y left, z up):
```python
p = prediction_input.lidar_points          # (N,3)
m = (p[:,0]>0.5) & (np.abs(p[:,1])<self._corridor_halfwidth) & (p[:,2]>-1.0) & (p[:,2]<2.5)  # forward, in-lane, above ground below roof
# per azimuth bin (by y within corridor): nearest x
```
Cap/steer as today. This is GT geometry → the clean upper bound.

### 7. Rebuild + test
```
cd src/grpc && uv run compile-protos
VIRTUAL_ENV=../.venv uv pip install -e ./src/grpc ./src/runtime ./src/driver   # or reinstall touched pkgs
```
Then run `run_local_occ_da3.sh` variant with `driver.model.occ_mode=lidar` (+ `runtime.simulation_config.
send_lidar=true`). Sanity: lidar points arrive (log N per step), obstacle scenes show near points, clear
scenes don't. Then re-run the multi-scene ablation: ego-only vs +occ(lidar-GT).

## Risks / notes
- The render-event abstraction (step 3) is the trickiest — must submit lidar in lock-step with the camera
  frame the driver predicts on, else poses desync.
- `render_lidar` return format: check `point_xyzs` (repeated float) vs `point_xyzs_buffer` (bytes) in the
  actual sensorsim response; AV2 recon must support lidar rendering (thesis reconstructed with up_lidar, so
  the geometry supports it — but confirm the sensorsim `render_lidar` works on these usdz).
- Keep it behind a flag so the camera-only path (M0/M1a) stays intact.
