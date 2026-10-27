#!/bin/sh
# Export the MicroCluck toe STLs from the OpenSCAD sources.
#
#   hardware/tools/build_toes.sh [out_dir] [sim_assets_dir]
#
# out_dir          STLs, ASCII (default: hardware/build)
# sim_assets_dir   if given, binary copies of the two simulation meshes are also written there
#                  (MuJoCo only reads binary STL)
#
# Written: toes_<side>.stl (the toes alone, simulation visual), collision_with_toes_<side>.stl (convex hull of
# Pollen's sole plus the toes, simulation collision) and sole_with_toes_<side>.stl (the printable part). The last
# two need a local copy of Pollen's assets (POLLEN_ASSETS, default ~/microduck_rl/.../microduck/assets).
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$here/build"}
sim=${2:-}
pollen=${POLLEN_ASSETS:-"$HOME/microduck_rl/src/mjlab_microduck/robot/microduck/assets"}
mkdir -p "$out"
[ -n "$sim" ] && mkdir -p "$sim"

to_sim() { [ -n "$sim" ] && python3 "$here/tools/stl_ascii_to_binary.py" "$1" "$sim/$2" || true; }

for side in left right; do
  openscad -D "side=\"$side\"" -o "$out/toes_$side.stl" "$here/toes.scad" 2>&1 | grep -E "ERROR|WARNING" || true
  to_sim "$out/toes_$side.stl" "toes_$side.stl"
  if [ -f "$pollen/sole_$side.stl" ]; then
    openscad -D "side=\"$side\"" -D "pollen_dir=\"$pollen\"" -o "$out/collision_with_toes_$side.stl" "$here/collision_with_toes.scad" 2>&1 \
      | grep -E "ERROR|WARNING" || true
    to_sim "$out/collision_with_toes_$side.stl" "collision_with_toes_$side.stl"
    openscad -D "side=\"$side\"" -D "pollen_dir=\"$pollen\"" -o "$out/sole_with_toes_$side.stl" "$here/sole_with_toes.scad" 2>&1 \
      | grep -E "ERROR|WARNING|Simple:" || true
  else
    echo "no Pollen assets at $pollen: skipped the collision hull and sole_with_toes for $side"
  fi
done
ls -la "$out"/*.stl | awk '{print $5, $9}'
