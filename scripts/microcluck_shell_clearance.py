"""Does the body shell clear the legs? Signed clearance between the shell's outer envelope and every leg part.

    uv run python scripts/microcluck_shell_clearance.py --poses ~/stock_gait.npz

The shell envelope (hardware/shell.scad, solid outer shape) is convex, so it is a set of planes. For each pose
the convex-hull vertices of every leg mesh are moved into the trunk frame and tested against those planes:
clearance = the largest plane distance (positive outside, negative inside = penetration depth). For a point
outside a convex polytope the largest plane distance is a lower bound on the true distance, so reported
clearances are conservative; leg meshes are compared as convex hulls, which is conservative too.

Poses come from two places:
  - real gait: leg joint positions recorded while a trained walking policy ran (torque_baseline.py --dump-joints);
    Pollen's leg geometry is the same on MicroCluck, so their gait is a fair stand-in until MicroCluck walks
  - stress: random poses over most of each joint's mechanical range
"""

import argparse
import struct
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent / "src/mjlab_microduck/robot/microcluck"
LEG_BODIES = {"yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left",
              "bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right"}


def read_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
    return tri["v"].reshape(-1, 3).astype(float)


def envelope_planes(path: Path):
    verts = read_binary_stl(path) / 1000.0  # mm -> m, trunk frame
    hull = ConvexHull(verts)
    eq = hull.equations  # n.x + c <= 0 inside
    return eq[:, :3], eq[:, 3]


class Sweep:
    def __init__(self, scene: Path, envelope: Path):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.normals, self.offsets = envelope_planes(envelope)
        m = self.m
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        self.joint_names = [n for n in names if n and n != "trunk_base_freejoint"]
        self.jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names}
        self.adr = [m.jnt_qposadr[self.jid[n]] for n in self.joint_names]
        self.geoms = []  # (geom id, body name, hull vertices in the geom frame)
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            if body in LEG_BODIES and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                self.geoms.append((g, body, v[ConvexHull(v).vertices]))
        self.data = mujoco.MjData(m)

    def clearance(self, q):
        """Smallest signed clearance (m) over all leg hull vertices, and the body it belongs to."""
        m, d = self.m, self.data
        for adr, v in zip(self.adr, q):
            d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        origin, rot = d.xpos[self.trunk], d.xmat[self.trunk].reshape(3, 3)
        best, who = 1e9, ""
        for g, body, pts in self.geoms:
            world = pts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            local = (world - origin) @ rot  # trunk frame
            sd = (local @ self.normals.T + self.offsets).max(axis=1)
            c = sd.min()
            if c < best:
                best, who = c, body
        return best, who

    def sweep(self, poses, label):
        mins, whos = [], []
        for q in poses:
            c, w = self.clearance(q)
            mins.append(c)
            whos.append(w)
        mins = np.array(mins)
        worst = int(np.argmin(mins))
        print(f"{label:10s} {len(poses):6d} poses | min {mins.min() * 1000:7.1f} mm (closest: {whos[worst]}) | "
              f"1st pct {np.percentile(mins, 1) * 1000:6.1f} mm | median {np.median(mins) * 1000:6.1f} mm | "
              f"intersecting: {int((mins < 0).sum())} ({100 * (mins < 0).mean():.1f}%)")
        return mins, worst


def contents_fit(envelope: Path, torso_scad: Path, wall_mm: float) -> None:
    """Do the battery and electronics blocks sit inside the shell cavity? Margin to the inner wall, in mm."""
    import re

    text = torso_scad.read_text()
    vec = lambda name: np.array([float(v) for v in re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]", text).group(1).split(",")])  # noqa: E731
    normals, offsets = envelope_planes(envelope)
    keep = normals[:, 0] < 0.999  # the shell is open at the front (the bezel closes it): no wall on the +x plane
    normals, offsets = normals[keep], offsets[keep]
    for key in ("battery", "electronics"):
        pos, size = vec(f"{key}_pos") / 1000.0, vec(f"{key}_size") / 1000.0
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size / 2 + pos
        outside = (corners @ normals.T + offsets).max(axis=1)  # positive outside the outer shell, negative inside
        margin = (-outside - wall_mm / 1000.0).min() * 1000  # distance from the block to the inner wall
        print(f"fit         {key:12s} margin to the inner wall {margin:+6.1f} mm "
              f"({'inside' if margin >= 0 else 'PUSHES THROUGH THE WALL'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torso", default=str(ROOT / "assets/torso.scad"), help="torso.scad copied by hardware/tools/build_shell.sh")
    ap.add_argument("--wall", type=float, default=1.6, help="shell wall thickness in mm (shell_wall in torso.scad)")
    ap.add_argument("--fit-only", action="store_true", help="only check the battery/electronics fit")
    ap.add_argument("--scene", default=str(ROOT / "scene_walk.xml"))
    ap.add_argument("--envelope", default=str(ROOT / "assets/shell_envelope.stl"))
    ap.add_argument("--poses", help=".npz from torque_baseline.py --dump-joints")
    ap.add_argument("--gait-samples", type=int, default=6000)
    ap.add_argument("--stress-samples", type=int, default=6000)
    ap.add_argument("--stress-fraction", type=float, default=0.7, help="fraction of each joint's range used for stress poses")
    a = ap.parse_args()

    if Path(a.torso).exists():
        contents_fit(Path(a.envelope), Path(a.torso), a.wall)
    if a.fit_only:
        return
    s = Sweep(Path(a.scene), Path(a.envelope))
    print(f"{len(s.geoms)} leg meshes, {len(s.normals)} shell planes, envelope {Path(a.envelope).name}")
    d = mujoco.MjData(s.m)
    mujoco.mj_resetDataKeyframe(s.m, d, mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    home = [d.qpos[adr] for adr in s.adr]
    c, who = s.clearance(home)
    print(f"home pose   closest {c * 1000:.1f} mm ({who})")

    rng = np.random.default_rng(0)
    if a.poses:
        z = np.load(a.poses, allow_pickle=True)
        names_in = [str(n) for n in z["joint_names"]]
        cols = [names_in.index(n) for n in s.joint_names]
        pos = z["joint_pos"]
        pick = rng.choice(len(pos), size=min(a.gait_samples, len(pos)), replace=False)
        s.sweep(pos[pick][:, cols], "real gait")
    lo = np.array([s.m.jnt_range[s.jid[n]][0] for n in s.joint_names])
    hi = np.array([s.m.jnt_range[s.jid[n]][1] for n in s.joint_names])
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * a.stress_fraction
    stress = mid + rng.uniform(-1, 1, size=(a.stress_samples, len(s.joint_names))) * half
    mins, worst = s.sweep(stress, "stress")
    print("worst stress pose:", {n: round(float(v), 2) for n, v in zip(s.joint_names, stress[worst])})


if __name__ == "__main__":
    main()
