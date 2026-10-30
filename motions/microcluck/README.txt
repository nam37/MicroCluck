Open-loop recovery motions for the final hips-back MicroCluck (robot_standup_back29_fix_fan30.xml, MICROCLUCK_STAND_TRIM=0.258849,
--z-offset 0.07), found with scripts/standup_search.py. Replay with, for example:

  uv run python scripts/standup_search.py --play motions/microcluck/prone_side.pt --n 256 --use-mean

prone_back.pt   roll onto the front from flat on the back (8 keyframes, 6 s, all ten joints free); 0-30 deg from flat: 128/128 each
prone_side.pt   roll onto the front from a side (same search); 60-90 deg from flat: 128/128 each
chain_side.pt   the stand-up (6 keyframes, 5 s, mirrored legs) searched with prone_side.pt played first (--prefix); from a side: 254/256

The whole chain with the walking policy afterwards: scripts/render_chain_video.py.
