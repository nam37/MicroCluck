Open-loop recovery motions for the final hips-back MicroCluck (robot_standup_back29_fix_fan30.xml, MICROCLUCK_STAND_TRIM=0.258849,
--z-offset 0.07), found with scripts/standup_search.py. Replay with, for example:

  uv run python scripts/standup_search.py --play motions/microcluck/prone_side.pt --n 256 --use-mean

prone_back.pt   roll onto the front from flat on the back (8 keyframes, 6 s, all ten joints free); 0-30 deg from flat: 128/128 each
prone_side.pt   roll onto the front from a side (same search); 60-90 deg from flat: 128/128 each
chain_side.pt   the stand-up (6 keyframes, 5 s, mirrored legs) searched with prone_side.pt played first (--prefix); from a side: 254/256

The whole chain with the walking policy afterwards: scripts/render_chain_video.py.

Robust versions (searched with the physics randomized per robot, every candidate simulated in 4 robots and scored on their mean):

prone_side_r.pt, prone_back_r.pt   the rolls onto the front, from scratch: friction 0.3-1.5 on the floor and every part, battery 6.4-8.2 V
                                   with sag 0-0.2 V/Nm, servo delay 3-10 steps (15-50 ms), 0-60 g extra in the trunk, trunk centre of mass
                                   +-8 mm (standup_search.py --stress-* --stress-com-range --replicas 4)
chain_side_w.pt, chain_back_w.pt   the stand-ups behind those rolls, warm-started from chain_side.pt / chain_back.pt (--init, sigma 0.15,
                                   30 generations) over a moderate range: friction 0.4-1.4, battery 6.4-8.2 V, delay 3-8 steps,
                                   0-40 g, centre of mass +-5 mm

A from-scratch stand-up search over the wide range never reached standing (it settles on flinging the robot upright for an instant, or on
a motion that is upright only at the end of the window), which is why the stand-ups are warm-started. Use with the same model and
MICROCLUCK_STAND_TRIM as above; scripts/render_chain_video.py takes --prefix / --standup for the robust pair.
