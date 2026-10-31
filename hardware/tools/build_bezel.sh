#!/bin/sh
# Export the MicroCluck bezel STLs from the OpenSCAD source.
#
#   hardware/tools/build_bezel.sh [out_dir] [sim_assets_dir] [name=value ...]
#
# out_dir          bezel.stl (the frame with its two lips: the printable shape) and bezel_lips.stl (the lips alone), ASCII
# sim_assets_dir   if given, a copy of torso.scad goes there (the sim's lip clearance check and model test read it); the sim itself
#                  collides with primitives built from the same numbers, so it needs no STL. Pass "" to skip it.
# name=value       any torso.scad parameter to override for this build, e.g. bezel_lip_proud=8
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$here/build"}
sim=${2:-}
if [ $# -gt 2 ]; then shift 2; else shift $#; fi
mkdir -p "$out"
[ -n "$sim" ] && mkdir -p "$sim"

n=$#
while [ "$n" -gt 0 ]; do a=$1; shift; set -- "$@" -D "$a"; n=$((n - 1)); done

openscad "$@" -o "$out/bezel.stl" "$here/bezel.scad" 2>&1 | grep -E "ERROR|WARNING|Simple:" || true
rm -f "$out/bezel_lips.stl"
openscad "$@" -D 'part="lips"' -o "$out/bezel_lips.stl" "$here/bezel.scad" 2>&1 | grep -E "ERROR" || true

if [ -n "$sim" ]; then
  cp "$here/torso.scad" "$sim/torso.scad"
  while [ $# -ge 2 ]; do
    a=$2
    shift 2
    n=${a%%=*}
    v=${a#*=}
    sed -i "s|^$n\([[:space:]]*\)=[^;]*;|$n\1= $v;|" "$sim/torso.scad"
  done
fi
ls -la "$out"/bezel*.stl 2>/dev/null | awk '{print $5, $9}'
