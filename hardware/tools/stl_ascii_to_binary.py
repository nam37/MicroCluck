#!/usr/bin/env python3
"""Convert ASCII STL (what OpenSCAD 2021.01 writes) to binary STL, which MuJoCo requires.

    python3 stl_ascii_to_binary.py in.stl out.stl
"""
import struct
import sys


def convert(src: str, dst: str) -> int:
    verts = []
    with open(src) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "vertex":
                verts.append(tuple(float(x) for x in parts[1:4]))
    if len(verts) % 3:
        raise SystemExit(f"{src}: vertex count {len(verts)} is not a multiple of 3")
    with open(dst, "wb") as out:
        out.write(b"MicroCluck foot".ljust(80, b"\0"))
        out.write(struct.pack("<I", len(verts) // 3))
        for i in range(0, len(verts), 3):
            a, b, c = verts[i:i + 3]
            u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
            out.write(struct.pack("<12fH", n[0] / length, n[1] / length, n[2] / length, *a, *b, *c, 0))
    return len(verts) // 3


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"{sys.argv[2]}: {convert(sys.argv[1], sys.argv[2])} triangles")
