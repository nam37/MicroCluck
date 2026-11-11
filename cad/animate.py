"""Animates the whole MicroPup with the CAD legs: stand, trot, give a paw (abduction), crouch, stand.

Kinematic only (no physics): each part is a MuJoCo mocap body placed every frame from the same linkage maths as
leg.py (the knee crank, links and bellcrank follow the knee). Meshes and constants come from anim_export.py.

Usage (in WSL, with MuJoCo): python animate.py OUT.mp4 [--frames a,b,c]   (--frames writes PNG stills instead)
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
A = HERE / "anim"
C = json.loads((A / "consts.json").read_text())
FPS, W, H = 30, 960, 540

COLORS = {
    "holder": (0.91, 0.90, 0.87), "cover": (0.95, 0.94, 0.92), "abd_mount": (0.72, 0.70, 0.66),
    "outer_plate": (0.78, 0.80, 0.83), "inner_plate": (0.78, 0.80, 0.83), "bridge": (0.66, 0.68, 0.72),
    "shank": (0.78, 0.80, 0.83), "foot": (0.93, 0.70, 0.08),
    "bellcrank": (0.30, 0.52, 0.84), "knee_crank": (0.30, 0.52, 0.84), "long_link": (0.86, 0.50, 0.15),
    "short_link": (0.86, 0.50, 0.15), "hip_servo": (0.13, 0.13, 0.14), "knee_servo": (0.13, 0.13, 0.14),
    "abd_servo": (0.13, 0.13, 0.14), "hip_horn": (0.96, 0.96, 0.96), "knee_horn": (0.96, 0.96, 0.96),
    "abd_horn": (0.96, 0.96, 0.96),
    "body": (0.95, 0.95, 0.93), "bezel": (0.96, 0.72, 0.05), "screen": (0.03, 0.03, 0.04),
    "eyes": (0.25, 0.85, 1.0), "tail": (0.12, 0.12, 0.13),
}
LEGS = {"fl": (1, 1), "fr": (1, -1), "rl": (-1, 1), "rr": (-1, -1)}   # (front?, left?) signs


def build_model():
    assets, bodies = [], []
    names = [(f"{p}_{v}", p) for p in C["parts"] for v in LEGS] + [(b, b) for b in C["body_parts"]]
    for mesh, _ in names:
        assets.append(f'<mesh name="{mesh}" file="{A / (mesh + ".stl")}" scale="0.001 0.001 0.001" inertia="shell"/>')
    for mesh, kind in names:
        r, g, b = COLORS[kind]
        bodies.append(f'<body name="{mesh}" mocap="true"><geom type="mesh" mesh="{mesh}" rgba="{r} {g} {b} 1" '
                      f'contype="0" conaffinity="0" mass="0.001"/></body>')
    xml = f"""<mujoco>
  <compiler angle="degree"/>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7" rgb2="0.08 0.12 0.2" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.25 0.38" rgb2="0.22 0.33 0.48" width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.15"/>
    {"".join(assets)}
  </asset>
  <worldbody>
    <light pos="0.4 0.6 1.2" dir="-0.3 -0.5 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom type="plane" size="3 3 0.01" material="grid" contype="0" conaffinity="0"/>
    {"".join(bodies)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


# ---- kinematics (leg frame, mm, degrees; mirrors leg.py) ----------------------------------------------------------
def T(x=0.0, y=0.0, z=0.0):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m


def Ry(deg):   # build123d Rot(0, deg, 0)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4); m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c; return m


def Rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4); m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c; return m


def pt(r, a, base=(0.0, 0.0)):
    return base[0] + r * math.cos(math.radians(a)), base[1] + r * math.sin(math.radians(a))


MX, MY = np.diag([-1.0, 1, 1, 1]), np.diag([1.0, -1, 1, 1])


def leg_poses(t, q, a):
    """{part: 4x4} in the front-left leg frame."""
    s = t + q
    phi = s + C["ALPHA"]
    psi = phi + C["BETA"]
    K = pt(C["L_THIGH"], t)
    B, L = pt(C["RB"], phi), pt(C["RB"], s + C["ALPHA"], K)
    P1 = pt(C["RS"], psi)
    link = math.degrees(math.atan2(L[1] - B[1], L[0] - B[0]))
    thigh = Ry(-(t + 90))
    shank = T(K[0], 0, K[1]) @ Ry(-(s + 90))
    ks = T(0, 0, C["KS_Z"])
    p = dict(holder=np.eye(4), cover=np.eye(4), hip_servo=np.eye(4), knee_servo=ks, abd_servo=np.eye(4),
             abd_horn=np.eye(4), abd_mount=np.eye(4), outer_plate=thigh, inner_plate=thigh, bridge=thigh,
             hip_horn=thigh, shank=shank, foot=shank, bellcrank=Ry(-phi), knee_crank=ks @ Ry(-psi),
             knee_horn=ks @ Ry(-psi), long_link=T(B[0], 0, B[1]) @ Ry(-link), short_link=T(P1[0], 0, P1[1]))
    abd = T(0, C["ABD_Y"], 0) @ Rx(a) @ T(0, -C["ABD_Y"], 0)
    return {k: (v if k in ("abd_servo", "abd_mount") else abd @ v) for k, v in p.items()}


def servo_angle(t, q):
    return (t + q + C["ALPHA"] + C["BETA"] - C["KNEE_MID"] + 180.0) % 360.0 - 180.0


def robot_poses(joints, body):
    """joints: {leg: (t, q, a)}; body: 4x4 of the body frame (hip plane at z = 0) in the world, mm. {mesh: 4x4}"""
    out = {b: body for b in C["body_parts"]}
    side = set(C["HOLDER_SIDE"])
    for leg, (front, left) in LEGS.items():
        t, q, a = joints[leg]
        base = body @ T(front * C["HIP_X"], left * C["LEG_Y"], 0)
        for part, m in leg_poses(t, q, a).items():
            if front < 0 and part in side:
                m = MX @ m @ MX
            if left < 0:
                m = MY @ m @ MY
            out[f"{part}_{leg}"] = base @ m
    return out


def apply(model, data, poses):
    for name, m in poses.items():
        i = model.body(name).mocapid[0]
        data.mocap_pos[i] = m[:3, 3] / 1000.0
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, m[:3, :3].flatten())
        data.mocap_quat[i] = quat


# ---- choreography ---------------------------------------------------------------------------------------------
TH, QH = C["T_HOME"], C["Q_HOME"]
GAIT_F, HIP_AMP, KNEE_LIFT = 2.0, 11.5, 23.0   # the sim's 2.5 Hz trot is 0.20 / 0.40 rad; a little slower here
STRIDE_V = 4 * GAIT_F * math.radians(HIP_AMP) * 75.0   # mm/s: a stance foot sweeps 2 x amplitude x 75 mm per half cycle
SEGMENTS = [("stand", 1.0), ("trot", 6.0), ("paw up", 0.6), ("paw", 2.4), ("paw down", 0.6), ("crouch", 1.0),
            ("hold", 0.6), ("rise", 1.0), ("end", 0.8)]
PAW = (TH + 60.0, 40.0)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def crouch_angles(h):
    """Thigh and knee with the foot straight under the hip at hip-to-foot-centre height h."""
    c = math.degrees(math.acos(h / (C["L_THIGH"] + C["L_SHANK"])))
    return -90.0 - c, 2 * c


def frame(time):
    """Joint angles, body x and body height (hip plane) at `time`, plus the segment name."""
    home = {leg: (TH, QH, 0.0) for leg in LEGS}
    h0 = 75.0 + C["FOOT_R"]
    t0 = 0.0
    x_body = 0.0
    for name, dur in SEGMENTS:
        if time < t0 + dur or name == "end":
            u = (time - t0) / dur
            break
        if name == "trot":
            x_body += STRIDE_V * (dur - 0.5)
        t0 += dur
    j = dict(home)
    z = h0
    if name == "trot":
        amp = smooth(u * dur / 0.5) * smooth((dur - u * dur) / 0.5)
        for leg in LEGS:
            ph = 2 * math.pi * GAIT_F * (time - t0) + (0.0 if leg in ("fl", "rr") else math.pi)
            j[leg] = (TH + amp * HIP_AMP * math.sin(ph), QH + amp * KNEE_LIFT * max(0.0, math.cos(ph)), 0.0)
        # integrate the body's travel at the stance feet's speed, scaled by the ramp
        n = 40
        x_body += sum(STRIDE_V * smooth(k * u * dur / n / 0.5) * smooth((dur - k * u * dur / n) / 0.5)
                      for k in range(n)) * (u * dur / n)
    elif name in ("paw up", "paw", "paw down"):
        k = smooth(u) if name == "paw up" else (1.0 if name == "paw" else 1 - smooth(u))
        a = 0.0
        if name == "paw":   # abduction swings out and back while the paw is up
            tau = u * dur
            a = smooth(tau / 0.4) * smooth((dur - tau) / 0.4) * (7.5 + 17.5 * math.sin(2 * math.pi * tau))
        j["fl"] = (TH + k * (PAW[0] - TH), QH + k * (PAW[1] - QH), a)
    elif name in ("crouch", "hold", "rise", "end"):
        k = smooth(u) if name == "crouch" else (1.0 if name == "hold" else (1 - smooth(u) if name == "rise" else 0.0))
        h = 75.0 - k * 27.0
        tc, qc = crouch_angles(h)
        j = {leg: (tc, qc, 0.0) for leg in LEGS}
        z = h + C["FOOT_R"]
    return j, x_body, z, name


def body_matrix(x, z):
    return T(x, 0, z)


def render(out, stills=None):
    import mediapy as media
    import imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    model = build_model()
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    total = sum(d for _, d in SEGMENTS)
    times = [s / FPS for s in (stills or range(int(total * FPS)))]
    frames, worst = [], 0.0
    for tm in times:
        j, xb, z, seg = frame(tm)
        worst = max(worst, max(abs(servo_angle(t, q)) for t, q, _ in j.values()))
        apply(model, data, robot_poses(j, body_matrix(xb, z)))
        mujoco.mj_forward(model, data)
        close = seg.startswith("paw")
        cam.lookat[:] = ((xb + (70 if close else 0)) / 1000, (30 if close else 0) / 1000, (z - 15) / 1000)
        cam.distance = 0.38 if close else 0.50
        cam.azimuth = 150 + 25 * math.sin(tm / total * 2 * math.pi)
        cam.elevation = -16
        r.update_scene(data, camera=cam)
        frames.append(r.render())
    print(f"{len(frames)} frames, knee servo at most {worst:.0f} deg from centre (usable {C['KNEE_WINDOW']:.0f})")
    if stills:
        for f, s in zip(frames, stills):
            media.write_image(str(Path(out).with_suffix("")) + f"_{s:04d}.png", f)
    else:
        media.write_video(out, frames, fps=FPS)
    print("wrote", out)


if __name__ == "__main__":
    out = sys.argv[1]
    st = None
    if "--frames" in sys.argv:
        st = [int(x) for x in sys.argv[sys.argv.index("--frames") + 1].split(",")]
    render(out, st)
