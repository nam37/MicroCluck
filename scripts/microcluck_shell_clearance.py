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


def _planes(verts: np.ndarray):
    eq = np.unique(np.round(ConvexHull(verts).equations, 7), axis=0)  # n.x + c <= 0 inside; merge identical facets
    return eq[:, :3], eq[:, 3]


def envelope_planes(path: Path):
    return _planes(read_binary_stl(path) / 1000.0)  # mm -> m, trunk frame


def wing_planes(path: Path):
    """The wing STL holds two separate discs, one on each side: a convex plane set for each (split by the sign of y)."""
    tri = read_binary_stl(path).reshape(-1, 3, 3) / 1000.0
    left = tri[tri[:, :, 1].mean(axis=1) > 0].reshape(-1, 3)
    right = tri[tri[:, :, 1].mean(axis=1) < 0].reshape(-1, 3)
    return [_planes(left), _planes(right)]


class Sweep:
    def __init__(self, scene: Path, envelope: Path, wings: Path | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.normals, self.offsets = envelope_planes(envelope)
        self.wing_shapes = wing_planes(wings) if wings is not None else []
        self.wing = None  # (clearance, body) to the wing discs for the last pose, when they are loaded
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
                hull = v[ConvexHull(v).vertices]
                centre = (hull.max(axis=0) + hull.min(axis=0)) / 2
                self.geoms.append((g, body, hull, centre, np.linalg.norm(hull - centre, axis=1).max()))
        self.data = mujoco.MjData(m)

    @staticmethod
    def _closest(shapes, geoms, cents, local_of):
        """Smallest signed clearance to the convex shapes over all leg parts, with the part it belongs to.

        Plane distances are 1-Lipschitz, so a part's clearance is at least its centre's distance minus its radius.
        Parts are visited in order of that bound and the search stops once the bound passes the best found, which
        skips the parts far from the shell and still returns the exact minimum."""
        best, who = 1e9, ""
        for normals, offsets in shapes:
            bound = np.array([(c @ normals.T + offsets).max() - rad for c, (*_, rad) in zip(cents, geoms)])
            for i in np.argsort(bound):
                if bound[i] >= best:
                    break
                c = (local_of(i) @ normals.T + offsets).max(axis=1).min()
                if c < best:
                    best, who = c, geoms[i][1]
        return best, who

    def clearance(self, q):
        """Smallest signed clearance (m) over all leg hull vertices, and the body it belongs to. With wing discs
        loaded, the same for the discs is left in self.wing."""
        m, d = self.m, self.data
        for adr, v in zip(self.adr, q):
            d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        origin, rot = d.xpos[self.trunk], d.xmat[self.trunk].reshape(3, 3)
        rots = [d.geom_xmat[g].reshape(3, 3) for g, *_ in self.geoms]
        poss = [d.geom_xpos[g] for g, *_ in self.geoms]
        cents = [((r @ c + p) - origin) @ rot for r, p, (_, _, _, c, _) in zip(rots, poss, self.geoms)]

        def local_of(i):  # a part's hull vertices in the trunk frame
            return ((self.geoms[i][2] @ rots[i].T + poss[i]) - origin) @ rot

        if self.wing_shapes:
            self.wing = self._closest(self.wing_shapes, self.geoms, cents, local_of)
        return self._closest([(self.normals, self.offsets)], self.geoms, cents, local_of)

    def sweep(self, poses, label):
        mins, whos, wmins = [], [], []
        for q in poses:
            c, w = self.clearance(q)
            mins.append(c)
            whos.append(w)
            if self.wing_shapes:
                wmins.append(self.wing[0])
        mins = np.array(mins)
        worst = int(np.argmin(mins))
        print(f"{label:10s} {len(poses):6d} poses | min {mins.min() * 1000:7.1f} mm (closest: {whos[worst]}) | "
              f"1st pct {np.percentile(mins, 1) * 1000:6.1f} mm | median {np.median(mins) * 1000:6.1f} mm | "
              f"intersecting: {int((mins < 0).sum())} ({100 * (mins < 0).mean():.1f}%)")
        if wmins:
            wmins = np.array(wmins)
            print(f"{label:10s} wing discs: min {wmins.min() * 1000:7.1f} mm | intersecting: {int((wmins < 0).sum())}")
        return mins, worst


def _scalar(text: str, name: str) -> float:
    import re

    return float(re.search(rf"\b{name}\s*=\s*(-?[0-9.]+)\s*;", text).group(1))


def contents_fit(envelope: Path, torso_scad: Path, wall_mm: float, cavity: Path | None = None) -> dict[str, float]:
    """Do the battery and electronics blocks sit inside the shell cavity? Margin to the inner wall, in mm.

    With the exported cavity (shell_cavity.stl) the blocks are tested against the real inner space, and the wall
    thickness is measured (smallest distance from a cavity vertex to the outer shell). Without it the cavity is
    assumed to be the envelope shrunk by wall_mm, which is only right when the wall is uniform."""
    import re

    results: dict[str, float] = {}
    text = torso_scad.read_text()
    vec = lambda name: np.array([float(v) for v in re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]", text).group(1).split(",")])  # noqa: E731
    outer_n, outer_c = envelope_planes(envelope)
    keep = outer_n[:, 0] < 0.999  # the shell is open at the front (the bezel closes it): no wall on the +x plane
    outer_n, outer_c = outer_n[keep], outer_c[keep]
    if cavity is not None:
        normals, offsets = envelope_planes(cavity)
        keep = normals[:, 0] < 0.999
        normals, offsets, wall_mm = normals[keep], offsets[keep], 0.0
        cav = read_binary_stl(cavity) / 1000.0
        wall = (-(cav @ outer_n.T + outer_c).max(axis=1)).min() * 1000
        results["wall"] = wall
        print(f"wall        thinnest {wall:.2f} mm (nominal {_scalar(text, 'shell_wall'):.1f} mm in torso.scad)")
    else:
        normals, offsets = outer_n, outer_c
    keys = ["battery", "electronics"]
    rear = re.search(r"\bshell_rear_x\s*=\s*(-?[0-9.]+)\s*;", text)  # older torso.scad files have no such line
    if rear and float(rear.group(1)) < vec("tail_pos")[0] + vec("tail_size")[0] / 2 - 1:
        keys.append("tail")  # the body runs back past the tail block's front face, so the block has to fit inside too
    for key in keys:
        pos, size = vec(f"{key}_pos") / 1000.0, vec(f"{key}_size") / 1000.0
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size / 2 + pos
        outside = (corners @ normals.T + offsets).max(axis=1)  # positive outside the outer shell, negative inside
        margin = (-outside - wall_mm / 1000.0).min() * 1000  # distance from the block to the inner wall
        results[key] = margin
        print(f"fit         {key:12s} margin to the inner wall {margin:+6.1f} mm "
              f"({'inside' if margin >= 0 else 'PUSHES THROUGH THE WALL'})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torso", default=str(ROOT / "assets/torso.scad"), help="torso.scad copied by hardware/tools/build_shell.sh")
    ap.add_argument("--wall", type=float, default=1.6, help="shell wall thickness in mm (shell_wall in torso.scad)")
    ap.add_argument("--fit-only", action="store_true", help="only check the battery/electronics fit")
    ap.add_argument("--scene", default=str(ROOT / "scene_walk.xml"))
    ap.add_argument("--envelope", default=str(ROOT / "assets/shell_envelope.stl"))
    ap.add_argument("--cavity", default=str(ROOT / "assets/shell_cavity.stl"),
                    help="the exported inner space; when it exists the fit is tested against it and the wall is measured")
    ap.add_argument("--wings", default=str(ROOT / "assets/wings.stl"),
                    help="wing discs (egg style); when they exist the clearance to each disc is reported too")
    ap.add_argument("--poses", help=".npz from torque_baseline.py --dump-joints")
    ap.add_argument("--gait-samples", type=int, default=6000)
    ap.add_argument("--stress-samples", type=int, default=6000)
    ap.add_argument("--stress-fraction", type=float, default=0.7, help="fraction of each joint's range used for stress poses")
    ap.add_argument("--part", type=int, default=0, help="with --of: check only every --of-th recorded pose, starting here")
    ap.add_argument("--of", type=int, default=1, help="split the recorded poses over this many runs (take the smallest min)")
    a = ap.parse_args()

    if Path(a.torso).exists():
        contents_fit(Path(a.envelope), Path(a.torso), a.wall, Path(a.cavity) if Path(a.cavity).exists() else None)
    if a.fit_only:
        return
    s = Sweep(Path(a.scene), Path(a.envelope), Path(a.wings) if Path(a.wings).exists() else None)
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
        pos = z["joint_pos"][a.part::a.of]
        pick = rng.choice(len(pos), size=min(a.gait_samples, len(pos)), replace=False)
        s.sweep(pos[pick][:, cols], "real gait")
    if a.stress_samples > 0:
        lo = np.array([s.m.jnt_range[s.jid[n]][0] for n in s.joint_names])
        hi = np.array([s.m.jnt_range[s.jid[n]][1] for n in s.joint_names])
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * a.stress_fraction
        stress = mid + rng.uniform(-1, 1, size=(a.stress_samples, len(s.joint_names))) * half
        mins, worst = s.sweep(stress, "stress")
        print("worst stress pose:", {n: round(float(v), 2) for n, v in zip(s.joint_names, stress[worst])})


if __name__ == "__main__":
    main()
