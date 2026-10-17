"""
Closed-loop vision-driven pick, MuJoCo, single file.

    render camera -> HSV blob detection -> deproject to world -> IK -> pick

Ground truth object position is read in exactly one place: to SCORE the run at
the end. It never drives the arm.

USAGE (run these in order the first time):
    python vision_pick_all.py --selftest        # pure math checks, no MuJoCo needed
    python vision_pick_all.py --check            # loads the model, renders once
    python vision_pick_all.py --runs 2            # the actual demo
    python vision_pick_all.py --runs 2 --view      # same, with a live 3-D window
    python vision_pick_all.py --runs 2 --record    # same, saves out/simulation.mp4
                                                    # (no window needed -- works headless)

Only needs: pip install mujoco opencv-python numpy
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# mujoco is imported lazily (inside functions that need it) so --selftest can
# run and prove the camera/IK math is correct even before mujoco is installed.


# ============================================================================
# 1. the scene -- table, 4-DOF arm + parallel gripper, fixed camera, red cube
# ============================================================================
SCENE_XML = r"""
<mujoco model="vision_pick">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.28 0.28 0.32" rgb2="0.34 0.34 0.38" width="300" height="300"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="table_mat" rgba="0.72 0.68 0.60 1"/>
    <material name="arm_mat"   rgba="0.26 0.30 0.38 1"/>
    <material name="cube_mat"  rgba="0.85 0.08 0.08 1"/>
  </asset>

  <default>
    <geom condim="4" friction="1 0.02 0.001" solref="0.005 1"/>
    <joint damping="2" armature="0.02"/>
    <default class="arm">
      <geom type="capsule" material="arm_mat" density="600" contype="2" conaffinity="1"/>
      <position kp="400" kv="40" forcerange="-200 200"/>
    </default>
    <default class="pad">
      <geom type="box" rgba="0.12 0.12 0.15 1" friction="2.0 0.05 0.0005" condim="4" density="800" contype="2" conaffinity="1"/>
      <position kp="600" kv="20" forcerange="-80 80"/>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.3 2.0" dir="0 0 -1" directional="true" diffuse="0.75 0.75 0.75"/>
    <light pos="-0.5 0.6 1.5" dir="0.4 -0.4 -1" diffuse="0.30 0.30 0.30"/>

    <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
    <geom name="table" type="box" pos="0.25 0 0.20" size="0.45 0.45 0.20" material="table_mat"/>

    <!-- near-overhead (75 deg below horizontal): a shallower angle lets a real
         3-D cube's visible side face bias the detected centroid toward the camera -->
    <camera name="cam" pos="0.36 -0.15 1.15" xyaxes="1 0 0  0 0.9659 0.2588" fovy="40"/>

    <!-- third-person camera for recording video. It auto-tracks the cube, so it
         frames each run well no matter where the object spawns. -->
    <camera name="watch" pos="0.95 -0.95 0.85" mode="targetbody" target="cube" fovy="45"/>

    <body name="base" pos="0 0 0.40">
      <geom class="arm" type="cylinder" size="0.06 0.03" pos="0 0 0.03"/>

      <body name="link1" pos="0 0 0.06" gravcomp="1">
        <joint name="j1" axis="0 0 1" range="-3.1 3.1"/>
        <geom class="arm" fromto="0 0 0  0 0 0.12" size="0.040"/>

        <body name="link2" pos="0 0 0.12" gravcomp="1">
          <joint name="j2" axis="0 1 0" range="-2.6 2.6"/>
          <geom class="arm" fromto="0 0 0  0 0 0.28" size="0.035"/>

          <body name="link3" pos="0 0 0.28" gravcomp="1">
            <joint name="j3" axis="0 1 0" range="-2.6 2.6"/>
            <geom class="arm" fromto="0 0 0  0 0 0.25" size="0.030"/>

            <body name="hand" pos="0 0 0.25" gravcomp="1">
              <joint name="j4" axis="0 1 0" range="-3.1 3.1"/>
              <geom class="arm" fromto="0 0 0  0 0 0.06" size="0.026"/>
              <site name="ee" pos="0 0 0.115" size="0.006" rgba="0 1 0 1"/>

              <body name="finger_l" pos="-0.025 0 0.075" gravcomp="1">
                <joint name="fl" type="slide" axis="-1 0 0" range="-0.015 0.045"/>
                <geom name="pad_l" class="pad" size="0.006 0.018 0.040" pos="0 0 0.040"/>
              </body>
              <body name="finger_r" pos="0.025 0 0.075" gravcomp="1">
                <joint name="fr" type="slide" axis="1 0 0" range="-0.015 0.045"/>
                <geom name="pad_r" class="pad" size="0.006 0.018 0.040" pos="0 0 0.040"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="cube" pos="0.35 0 0.42">
      <joint type="free" name="cube_joint" damping="0" armature="0"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" material="cube_mat"
            density="400" friction="1.5 0.05 0.0005" condim="4"/>
    </body>
  </worldbody>

  <actuator>
    <position class="arm" name="a1"  joint="j1" ctrlrange="-3.1 3.1"/>
    <position class="arm" name="a2"  joint="j2" ctrlrange="-2.6 2.6"/>
    <position class="arm" name="a3"  joint="j3" ctrlrange="-2.6 2.6"/>
    <position class="arm" name="a4"  joint="j4" ctrlrange="-3.1 3.1"/>
    <position class="pad" name="afl" joint="fl" ctrlrange="-0.015 0.045"/>
    <position class="pad" name="afr" joint="fr" ctrlrange="-0.015 0.045"/>
  </actuator>

  <keyframe>
    <key name="home"
         qpos="0 -0.6 1.2 0  0.030 0.030  0.35 0 0.42 1 0 0 0"
         ctrl="0 -0.6 1.2 0  0.030 0.030"/>
  </keyframe>
</mujoco>
"""

TABLE_TOP = 0.40
CUBE_HALF = 0.02
CUBE_TOP = TABLE_TOP + 2 * CUBE_HALF
GRASP_Z = CUBE_TOP + 0.005
APPROACH_DZ = 0.14
SPAWN_X = (0.28, 0.44)
SPAWN_Y = (-0.15, 0.15)
GRIP_OPEN = 0.030
GRIP_CLOSED = -0.015
DETECT_TOL = 0.030
LIFT_TOL = 0.05
ARM_JOINTS = ("j1", "j2", "j3", "j4")
OUT = Path(__file__).with_name("out")


# ============================================================================
# 2. vision -- HSV blob detection + pixel -> world deprojection
# ============================================================================
@dataclass
class CameraParams:
    pos: np.ndarray
    mat: np.ndarray
    fovy_deg: float
    width: int
    height: int

    @property
    def focal_px(self):
        return 0.5 * self.height / np.tan(0.5 * np.deg2rad(self.fovy_deg))

    @property
    def center(self):
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0


def pixel_to_ray(cam, u, v):
    cx, cy = cam.center
    f = cam.focal_px
    d_cam = np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    d_world = cam.mat @ d_cam
    return cam.pos.copy(), d_world / np.linalg.norm(d_world)


def pixel_to_plane(cam, u, v, plane_z):
    origin, d = pixel_to_ray(cam, u, v)
    if abs(d[2]) < 1e-9:
        raise ValueError("viewing ray is parallel to the plane")
    t = (plane_z - origin[2]) / d[2]
    if t <= 0:
        raise ValueError("plane is behind the camera")
    return origin + t * d


def world_to_pixel(cam, p_world):
    p_cam = cam.mat.T @ (np.asarray(p_world, float) - cam.pos)
    if p_cam[2] >= 0:
        raise ValueError("point is behind the camera")
    z = -p_cam[2]
    cx, cy = cam.center
    f = cam.focal_px
    return cx + f * p_cam[0] / z, cy - f * p_cam[1] / z


RED_BANDS = (((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255)))


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_BANDS:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_object(rgb, min_area=60):
    mask = red_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask
    m = cv2.moments(c)
    uv = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return {"uv": uv, "area": area, "contour": c}, mask


def locate_object(rgb, cam, plane_z):
    det, mask = detect_object(rgb)
    if det is None:
        return None, mask
    det["world"] = pixel_to_plane(cam, det["uv"][0], det["uv"][1], plane_z)
    return det, mask


def annotate(rgb, det, cam=None, truth_xyz=None):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if det is not None:
        cv2.drawContours(img, [det["contour"]], -1, (0, 255, 0), 2)
        u, v = int(round(det["uv"][0])), int(round(det["uv"][1]))
        cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        x, y, z = det["world"]
        cv2.putText(img, f"detected  x={x:+.3f} y={y:+.3f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if truth_xyz is not None and cam is not None:
        tu, tv = world_to_pixel(cam, truth_xyz)
        cv2.drawMarker(img, (int(round(tu)), int(round(tv))), (255, 160, 0),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(img, f"truth     x={truth_xyz[0]:+.3f} y={truth_xyz[1]:+.3f}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    return img


# ============================================================================
# 3. inverse kinematics -- closed form + MuJoCo Jacobian refinement
# ============================================================================
@dataclass(frozen=True)
class ArmGeom:
    shoulder_z: float = 0.58
    l1: float = 0.28
    l2: float = 0.25
    l3: float = 0.115


GEOM = ArmGeom()


def forward_kinematics(q, g=GEOM):
    psi, a, b, c = q
    r = g.l1 * np.sin(a) + g.l2 * np.sin(a + b) + g.l3 * np.sin(a + b + c)
    z = g.shoulder_z + g.l1 * np.cos(a) + g.l2 * np.cos(a + b) + g.l3 * np.cos(a + b + c)
    return np.array([r * np.cos(psi), r * np.sin(psi), z])


def analytic_ik(target, g=GEOM):
    x, y, z = target
    psi = np.arctan2(y, x)
    r = np.hypot(x, y)
    a_pl, b_pl = r, (z + g.l3) - g.shoulder_z
    c = np.hypot(a_pl, b_pl)
    reach_max, reach_min = g.l1 + g.l2, abs(g.l1 - g.l2)
    ok = reach_min + 1e-6 < c < reach_max - 1e-6
    cos_beta = (c ** 2 - g.l1 ** 2 - g.l2 ** 2) / (2 * g.l1 * g.l2)
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    alpha = np.arctan2(a_pl, b_pl) - np.arctan2(g.l2 * np.sin(beta), g.l1 + g.l2 * np.cos(beta))
    gamma = np.pi - alpha - beta
    return np.array([psi, alpha, beta, gamma]), ok


def _arm_indices(model):
    import mujoco
    qadr, dadr = [], []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
    return np.array(qadr), np.array(dadr)


def ik_solve(model, data_scratch, target, q_seed=None, iters=60, tol=2e-4, damping=0.05):
    import mujoco
    target = np.asarray(target, float)
    qadr, dadr = _arm_indices(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    q, _ = analytic_ik(target)
    if q_seed is not None:
        q[0] = q[0] + np.round((q_seed[0] - q[0]) / (2 * np.pi)) * 2 * np.pi
    lo = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in ARM_JOINTS]][:, 0]
    hi = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in ARM_JOINTS]][:, 1]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    err = np.inf
    for _ in range(iters):
        data_scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, data_scratch)
        mujoco.mj_comPos(model, data_scratch)
        e = target - data_scratch.site_xpos[sid]
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        mujoco.mj_jacSite(model, data_scratch, jacp, jacr, sid)
        J = jacp[:, dadr]
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, e)
        q = np.clip(q + dq, lo, hi)
    return q, err


def fk_error(model, data_scratch, q):
    import mujoco
    qadr, _ = _arm_indices(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    data_scratch.qpos[qadr] = q
    mujoco.mj_kinematics(model, data_scratch)
    return float(np.linalg.norm(forward_kinematics(q) - data_scratch.site_xpos[sid]))


# ============================================================================
# 4. the environment -- wraps MuJoCo: reset, render, Cartesian move, gripper
# ============================================================================
class PickEnv:
    def __init__(self, width=640, height=480, view=False):
        if "MUJOCO_GL" not in os.environ and not (os.environ.get("DISPLAY") or os.name == "nt"):
            os.environ["MUJOCO_GL"] = "egl"
        import mujoco
        self._mj = mujoco

        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        nid = mujoco.mj_name2id
        O = mujoco.mjtObj
        self.cam_id = nid(self.model, O.mjOBJ_CAMERA, "cam")
        self.ee_id = nid(self.model, O.mjOBJ_SITE, "ee")
        self.cube_bid = nid(self.model, O.mjOBJ_BODY, "cube")
        self.cube_gid = nid(self.model, O.mjOBJ_GEOM, "cube")
        self.pad_gids = [nid(self.model, O.mjOBJ_GEOM, n) for n in ("pad_l", "pad_r")]
        self.cube_qadr = self.model.jnt_qposadr[nid(self.model, O.mjOBJ_JOINT, "cube_joint")]
        self.cube_dadr = self.model.jnt_dofadr[nid(self.model, O.mjOBJ_JOINT, "cube_joint")]
        self.arm_qadr = np.array([self.model.jnt_qposadr[nid(self.model, O.mjOBJ_JOINT, j)]
                                  for j in ARM_JOINTS])
        self.key_home = nid(self.model, O.mjOBJ_KEY, "home")

        self.viewer = None
        if view:
            import mujoco.viewer as mjv
            self.viewer = mjv.launch_passive(self.model, self.data)
        self._t0 = time.perf_counter()
        self._sim_t = 0.0

        self._record_writer = None
        self._record_camera = "watch"
        self._record_stride = 15   # capture every Nth physics step
        self._nstep = 0

    def start_recording(self, path, camera="watch", stride=15):
        """Write an MP4 of the simulation from `camera`, sampled every `stride`
        physics steps. Runs regardless of --view -- no window needed."""
        fps = 1.0 / (self.model.opt.timestep * stride)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._record_writer = cv2.VideoWriter(str(path), fourcc, fps, (self.width, self.height))
        if not self._record_writer.isOpened():
            self._record_writer = None
            raise RuntimeError(f"could not open a video writer for {path}")
        self._record_camera = camera
        self._record_stride = max(1, stride)
        self._nstep = 0

    def stop_recording(self):
        if self._record_writer is not None:
            self._record_writer.release()
            self._record_writer = None

    def close(self):
        self.stop_recording()
        if self.viewer is not None:
            self.viewer.close()

    def reset(self, rng):
        self._mj.mj_resetDataKeyframe(self.model, self.data, self.key_home)
        x, y = rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)
        yaw = rng.uniform(-np.pi, np.pi)
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [x, y, TABLE_TOP + CUBE_HALF + 1e-3]
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[self.cube_dadr:self.cube_dadr + 6] = 0
        self._mj.mj_forward(self.model, self.data)
        self._settle(300)
        return self.cube_pos

    def _settle(self, n):
        for _ in range(n):
            self._mj.mj_step(self.model, self.data)
            self._nstep += 1
            if self._record_writer is not None and self._nstep % self._record_stride == 0:
                self.renderer.update_scene(self.data, camera=self._record_camera)
                frame = self.renderer.render()
                self._record_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if self.viewer is not None:
                self._sim_t += self.model.opt.timestep
                self.viewer.sync()
                # pace to roughly real time so it's watchable, but never sleep
                # a negative amount if the physics itself is running slow
                target_wall = self._t0 + self._sim_t
                dt = target_wall - time.perf_counter()
                if dt > 0:
                    time.sleep(min(dt, 0.01))

    @property
    def q_arm(self):
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def ee_pos(self):
        return self.data.site_xpos[self.ee_id].copy()

    @property
    def cube_pos(self):
        return self.data.xpos[self.cube_bid].copy()

    def pads_touching_cube(self):
        touched = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.cube_gid in pair:
                for g in self.pad_gids:
                    if g in pair:
                        touched.add(g)
        return len(touched)

    def contact_report(self):
        """Every active contact right now, by geom name, with world position
        and penetration depth -- for diagnosing why the arm stopped short."""
        rows = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
            n2 = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
            rows.append((n1, n2, c.pos.copy(), float(c.dist)))
        return rows

    def render(self):
        self.renderer.update_scene(self.data, camera="cam")
        return self.renderer.render()

    def camera_params(self):
        return CameraParams(
            pos=self.data.cam_xpos[self.cam_id].copy(),
            mat=self.data.cam_xmat[self.cam_id].reshape(3, 3).copy(),
            fovy_deg=float(self.model.cam_fovy[self.cam_id]),
            width=self.width, height=self.height)

    def _hold(self, q, grip, steps):
        self.data.ctrl[:4] = q
        self.data.ctrl[4] = grip
        self.data.ctrl[5] = grip
        self._settle(steps)

    def set_gripper(self, grip, steps=250):
        self.data.ctrl[4] = grip
        self.data.ctrl[5] = grip
        self._settle(steps)

    def move_to(self, target_xyz, grip, n_way=40, sub=16, verbose=False):
        start = self.ee_pos
        target = np.asarray(target_xyz, float)
        q = self.q_arm
        worst = 0.0
        for s in np.linspace(0, 1, n_way)[1:]:
            wp = start + s * (target - start)
            q, err = ik_solve(self.model, self.scratch, wp, q_seed=q)
            worst = max(worst, err)
            self._hold(q, grip, sub)
        self._hold(q, grip, 120)
        if verbose:
            print(f"    worst IK residual over the path: {worst * 1000:.2f} mm")
        return np.linalg.norm(self.ee_pos - target)


# ============================================================================
# 5. the demo
# ============================================================================
def run_once(env, rng, idx):
    print(f"\n=== run {idx} " + "=" * 46)
    truth0 = env.reset(rng)
    print(f"  object placed at   x={truth0[0]:+.3f}  y={truth0[1]:+.3f}  (unknown to the controller)")

    rgb = env.render()
    cam = env.camera_params()
    det, mask = locate_object(rgb, cam, plane_z=CUBE_TOP)
    if det is None:
        print("  !! no red blob found")
        return {"run": idx, "ok": False, "why": "detection failed"}

    p = det["world"]
    err_xy = float(np.linalg.norm(p[:2] - truth0[:2]))
    print(f"  blob centroid      u={det['uv'][0]:7.2f}  v={det['uv'][1]:7.2f}  ({int(det['area'])} px)")
    print(f"  deprojected to     x={p[0]:+.3f}  y={p[1]:+.3f}")
    print(f"  perception error   {err_xy * 1000:.1f} mm")

    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / f"run{idx}_1_detection.png"), annotate(rgb, det, cam, truth0))
    cv2.imwrite(str(OUT / f"run{idx}_2_mask.png"), mask)

    gx, gy = float(p[0]), float(p[1])
    above = np.array([gx, gy, GRASP_Z + APPROACH_DZ])
    grasp = np.array([gx, gy, GRASP_Z])

    env.move_to(above, GRIP_OPEN, verbose=True)
    res = env.move_to(grasp, GRIP_OPEN, n_way=25, verbose=True)
    print(f"  tool point reached the grasp pose to within {res * 1000:.1f} mm")

    if res > 0.010:
        ee = env.ee_pos
        pad_l_z = float(env.data.geom_xpos[env.pad_gids[0]][2])
        pad_r_z = float(env.data.geom_xpos[env.pad_gids[1]][2])
        cube_now = env.cube_pos
        print(f"  !! DIAGNOSTIC: something is blocking the descent")
        print(f"     commanded grasp point  z={grasp[2]:.3f}   actual ee  z={ee[2]:.3f}"
              f"   (table top z={TABLE_TOP:.3f})")
        print(f"     pad heights now: left z={pad_l_z:.3f}  right z={pad_r_z:.3f}")
        print(f"     cube is currently at   x={cube_now[0]:+.3f} y={cube_now[1]:+.3f} z={cube_now[2]:.3f}"
              f"   (was x={truth0[0]:+.3f} y={truth0[1]:+.3f} z={truth0[2]:.3f} before the approach)")
        contacts = env.contact_report()
        if contacts:
            print(f"     active contacts ({len(contacts)}):")
            for n1, n2, cpos, dist in contacts:
                print(f"       {n1:>10s} <-> {n2:<10s}  at ({cpos[0]:+.3f},{cpos[1]:+.3f},{cpos[2]:+.3f})"
                      f"  penetration {-dist*1000:.1f} mm")
        else:
            print("     no active contacts (so it isn't touching anything -- likely a joint-limit "
                  "or reach problem, not a collision)")

    env.set_gripper(GRIP_CLOSED, steps=400)
    pads = env.pads_touching_cube()
    print(f"  pads in contact with the object: {pads}/2")

    env.move_to(np.array([gx, gy, GRASP_Z + 0.18]), GRIP_CLOSED, n_way=30)
    cv2.imwrite(str(OUT / f"run{idx}_3_lifted.png"), cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

    truth1 = env.cube_pos
    lift = float(truth1[2] - truth0[2])
    print(f"  object lifted      {lift * 100:.1f} cm")

    ok = err_xy < DETECT_TOL and lift > LIFT_TOL
    print("  RESULT:", "SUCCESS" if ok else "FAILED")
    return {"run": idx, "ok": ok, "truth": truth0[:2], "seen": p[:2],
            "err_mm": err_xy * 1000, "lift_cm": lift * 100, "pads": pads}


def cmd_demo(args):
    env = PickEnv(view=args.view)
    try:
        if args.record:
            OUT.mkdir(exist_ok=True)
            video_path = OUT / "simulation.mp4"
            env.start_recording(video_path)
            print(f"recording video to {video_path}")

        e = fk_error(env.model, env.scratch, np.array([0.3, 0.9, -0.4, 0.2]))
        print(f"analytic FK vs MuJoCo FK: {e * 1000:.3f} mm")

        rng = np.random.default_rng(args.seed)
        results = [run_once(env, rng, i + 1) for i in range(args.runs)]

        print("\n" + "=" * 60)
        print(f"{'run':>4} {'true x,y':>16} {'seen x,y':>16} {'err':>8} {'lift':>8}  result")
        for r in results:
            if not r["ok"] and "why" in r:
                print(f"{r['run']:>4} {'-':>16} {'-':>16} {'-':>8} {'-':>8}  {r['why']}")
                continue
            print(f"{r['run']:>4} "
                  f"{r['truth'][0]:+.3f},{r['truth'][1]:+.3f} "
                  f"  {r['seen'][0]:+.3f},{r['seen'][1]:+.3f} "
                  f"{r['err_mm']:6.1f}mm {r['lift_cm']:6.1f}cm  "
                  f"{'SUCCESS' if r['ok'] else 'FAILED'}")
        n_ok = sum(r["ok"] for r in results)
        print(f"\n{n_ok}/{len(results)} succeeded.  images in {OUT}/")

        if args.record:
            env.stop_recording()
            print(f"video saved to {video_path}")

        if args.view:
            print("close the viewer window to exit")
            while env.viewer is not None and env.viewer.is_running():
                env.viewer.sync()
                time.sleep(0.02)
    finally:
        env.close()


def cmd_check(args):
    env = PickEnv(view=args.view)
    try:
        print(f"model loaded: nq={env.model.nq} nv={env.model.nv} nu={env.model.nu}")
        e = fk_error(env.model, env.scratch, np.array([0.3, 0.9, -0.4, 0.2]))
        print(f"analytic FK vs MuJoCo FK: {e * 1000:.4f} mm  "
              f"({'ok' if e < 1e-3 else 'MISMATCH'})")
        truth = env.reset(np.random.default_rng(0))
        rgb = env.render()
        print(f"render ok: {rgb.shape} {rgb.dtype}")
        cam = env.camera_params()
        det, mask = locate_object(rgb, cam, plane_z=CUBE_TOP)
        if det is None:
            OUT.mkdir(exist_ok=True)
            cv2.imwrite(str(OUT / "check_fail.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"!! object not detected - see {OUT}/check_fail.png")
            return
        err = np.linalg.norm(det["world"][:2] - truth[:2])
        print(f"detected {int(det['area'])} px blob, world error {err * 1000:.1f} mm")
        print("\nlooks good - now run:  python vision_pick_all.py --runs 2")
    finally:
        env.close()


# ============================================================================
# 6. self-test -- pure numpy/cv2, no MuJoCo required. Run this first.
# ============================================================================
def _scene_camera():
    """Camera pose/fov exactly matching SCENE_XML's <camera name="cam">."""
    view_deg = 75.0
    a = np.radians(view_deg)
    x = np.array([1.0, 0, 0])
    y = np.array([0, np.sin(a), np.cos(a)])
    z = np.cross(x, y)
    mat = np.column_stack([x, y, z])
    return CameraParams(pos=np.array([0.36, -0.15, 1.15]), mat=mat,
                        fovy_deg=40.0, width=640, height=480)


def _cube_corners(center_xy, z_bottom=TABLE_TOP, half=CUBE_HALF):
    cx, cy = center_xy
    zc = z_bottom + half
    return np.array([[cx + sx * half, cy + sy * half, zc + sz * half]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


_FACES = {
    "top": ([1, 3, 7, 5], (0, 0, 1)), "bottom": ([0, 2, 6, 4], (0, 0, -1)),
    "+x": ([4, 5, 7, 6], (1, 0, 0)), "-x": ([0, 1, 3, 2], (-1, 0, 0)),
    "+y": ([2, 3, 7, 6], (0, 1, 0)), "-y": ([0, 1, 5, 4], (0, -1, 0)),
}


def _draw_cube(img, cam, center_xy, color=(217, 20, 20)):
    """Renders a REAL 3-D cube (top + visible side faces), not a flat rectangle
    -- catches the camera-angle side-face bias a flat-rectangle test would miss."""
    corners = _cube_corners(center_xy)
    visible = []
    for name, (idxs, normal) in _FACES.items():
        face_center = np.mean(corners[idxs], axis=0)
        if np.dot(face_center - cam.pos, normal) < 0:
            visible.append((np.linalg.norm(face_center - cam.pos), idxs))
    visible.sort(key=lambda t: -t[0])
    for _, idxs in visible:
        pts = np.array([world_to_pixel(cam, corners[i]) for i in idxs], np.int32)
        cv2.fillConvexPoly(img, pts, color)


def selftest():
    cam = _scene_camera()
    print("running pure-math self-test (no MuJoCo needed)...\n")

    # camera round-trip
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        p = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y), CUBE_TOP])
        u, v = world_to_pixel(cam, p)
        assert 0 <= u < cam.width and 0 <= v < cam.height, f"{p} off-image"
        back = pixel_to_plane(cam, u, v, CUBE_TOP)
        worst = max(worst, np.linalg.norm(back - p))
    print(f"  camera round-trip      : worst {worst*1e6:.3f} um, whole spawn box in frame")
    assert worst < 1e-6

    # IK round-trip over the reachable spawn envelope
    worst_fk, n = 0.0, 0
    for x in np.linspace(*SPAWN_X, 12):
        for y in np.linspace(*SPAWN_Y, 12):
            for z in (GRASP_Z, GRASP_Z + APPROACH_DZ, GRASP_Z + 0.18):
                q, ok = analytic_ik(np.array([x, y, z]))
                assert ok, f"target ({x:.2f},{y:.2f},{z:.2f}) unreachable"
                worst_fk = max(worst_fk, np.linalg.norm(forward_kinematics(q) - [x, y, z]))
                assert abs(q[0]) < 3.1 and all(abs(a) < 2.6 for a in q[1:3]) and abs(q[3]) < 3.1
                n += 1
    print(f"  analytic IK            : {n} targets, worst FK residual {worst_fk*1e9:.3f} nm, all in joint limits")
    assert worst_fk < 1e-6

    # realistic 3-D cube detection (this is the check that caught the real bug)
    errs = []
    rng = np.random.default_rng(3)
    for _ in range(200):
        img = np.full((cam.height, cam.width, 3), (184, 173, 153), np.uint8)
        p = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
        _draw_cube(img, cam, p)
        det, _ = detect_object(img)
        assert det is not None, "missed the cube"
        back = pixel_to_plane(cam, *det["uv"], CUBE_TOP)
        errs.append(np.linalg.norm(back[:2] - p))
    print(f"  3-D cube detection     : mean {np.mean(errs)*1000:.1f} mm, "
          f"worst {np.max(errs)*1000:.1f} mm  (tolerance is {DETECT_TOL*1000:.0f} mm)")
    assert np.max(errs) < DETECT_TOL

    # empty-scene safety
    det, mask = detect_object(np.full((120, 160, 3), (184, 173, 153), np.uint8))
    assert det is None and mask.sum() == 0
    print("  empty scene            : returns None, no phantom detection")

    print("\nall self-tests passed. Next: python vision_pick_all.py --check")


# ============================================================================
# 7. entry point
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="pure math checks, no MuJoCo needed")
    ap.add_argument("--check", action="store_true", help="load the model + render once")
    ap.add_argument("--runs", type=int, default=2, help="number of demo runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="open a live 3-D viewer window")
    ap.add_argument("--record", action="store_true",
                    help="save an MP4 of the run to out/simulation.mp4 (works with or without --view)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("MuJoCo isn't installed. Run:  pip install mujoco opencv-python numpy")
        sys.exit(1)

    if args.check:
        cmd_check(args)
    else:
        cmd_demo(args)


if __name__ == "__main__":
    main()