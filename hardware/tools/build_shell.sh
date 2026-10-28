#!/bin/sh
# Export the MicroCluck shell STLs from the OpenSCAD sources.
#
#   hardware/tools/build_shell.sh [out_dir] [sim_assets_dir] [style] [name=value ...]
#
# out_dir          shell.stl (hollow, printable), shell_envelope.stl (solid outer shape), shell_cavity.stl (inner
#                  space) and, for the egg style, wings.stl (the two wing discs), all ASCII
# sim_assets_dir   if given, binary copies are written there, plus a copy of torso.scad (the sim's test checks that
#                  file against its PARTS table). Pass "" to skip it.
# style            "egg" or "box"; default (or ""): shell_style in torso.scad
# name=value       any torso.scad parameter to override for this build, e.g. shell_rear_x=-30 for the shorter shell
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$here/build"}
sim=${2:-}
style=${3:-}
if [ $# -gt 3 ]; then shift 3; else shift $#; fi
mkdir -p "$out"
[ -n "$sim" ] && mkdir -p "$sim"

# what is left in "$@" are the overrides; turn each one into a "-D name=value" pair
[ -n "$style" ] && set -- "shell_style=\"$style\"" "$@"
n=$#
while [ "$n" -gt 0 ]; do a=$1; shift; set -- "$@" -D "$a"; n=$((n - 1)); done

openscad "$@" -o "$out/shell.stl" "$here/shell.scad" 2>&1 | grep -E "ERROR|WARNING|Simple:" || true
for p in envelope:shell_envelope.stl cavity:shell_cavity.stl wings:wings.stl; do
  part=${p%%:*}
  file=${p##*:}
  rm -f "$out/$file"
  openscad "$@" -D "part=\"$part\"" -o "$out/$file" "$here/shell.scad" 2>&1 | grep -E "ERROR" || true
done

wings=no
[ -s "$out/wings.stl" ] && grep -q facet "$out/wings.stl" && wings=yes
[ "$wings" = no ] && rm -f "$out/wings.stl"

if [ -n "$sim" ]; then
  for f in shell shell_envelope shell_cavity; do
    python3 "$here/tools/stl_ascii_to_binary.py" "$out/$f.stl" "$sim/$f.stl"
  done
  if [ "$wings" = yes ]; then python3 "$here/tools/stl_ascii_to_binary.py" "$out/wings.stl" "$sim/wings.stl"; else rm -f "$sim/wings.stl"; fi
  cp "$here/torso.scad" "$sim/torso.scad"
fi
ls -la "$out"/shell*.stl "$out"/wings.stl 2>/dev/null | awk '{print $5, $9}'
