#!/bin/sh
# Export the MicroCluck shell STLs from the OpenSCAD sources.
#
#   hardware/tools/build_shell.sh [out_dir] [sim_assets_dir]
#
# out_dir          shell.stl (hollow, printable) and shell_envelope.stl (solid outer shape), ASCII
# sim_assets_dir   if given, binary copies of both are written there, plus a copy of torso.scad (the sim's test
#                  checks that file against its PARTS table)
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$here/build"}
sim=${2:-}
mkdir -p "$out"
[ -n "$sim" ] && mkdir -p "$sim"

openscad -o "$out/shell.stl" "$here/shell.scad" 2>&1 | grep -E "ERROR|WARNING|Simple:" || true
openscad -D 'envelope=true' -o "$out/shell_envelope.stl" "$here/shell.scad" 2>&1 | grep -E "ERROR|WARNING" || true
if [ -n "$sim" ]; then
  python3 "$here/tools/stl_ascii_to_binary.py" "$out/shell.stl" "$sim/shell.stl"
  python3 "$here/tools/stl_ascii_to_binary.py" "$out/shell_envelope.stl" "$sim/shell_envelope.stl"
  cp "$here/torso.scad" "$sim/torso.scad"
fi
ls -la "$out/shell.stl" "$out/shell_envelope.stl" | awk '{print $5, $9}'
