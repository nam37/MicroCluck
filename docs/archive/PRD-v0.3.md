> **Superseded.** This is the v0.3 Markdown draft, kept for history only. The current PRD is [PRD.html](../PRD.html) (v0.5). Section numbers and milestones differ.

# MicroCluck — Product Requirements Document

> Status: **Draft v0.3** · Owner: Neil · Date: 2026-10-27
>
> v0.2: named MicroCluck; chicken design elements (comb, tail, three-toed feet); face expression set, parametric face rig, IMU-stabilized eyes, sound personality.
> v0.3: servo bus interface uses a through-hole 74HC126 buffer on perfboard (§5.2).

---

## 1. Summary

**MicroCluck** is a ~25 cm bipedal walking robot chicken in the style of Pollen Robotics' **Microduck**. Its entire torso is a vertical, front-facing **Freenove ESP32-S3 3.5" touchscreen** (FNK0104 family) that shows an animated chicken face. It has a comb on top, tail feathers at the back and three-toed feet, but no separate head, neck or beak. Walking is driven by a reinforcement-learning policy trained in simulation (forked from Pollen's open-source `microduck_rl`) and executed **entirely on the ESP32-S3 at 50 Hz**. The robot is driven from a **phone browser over the robot's own Wi-Fi**. A claw on top of the torso is planned for v2.

**Key architectural decision:** single MCU. The ESP32-S3 runs the control loop, policy inference, servo bus, IMU fusion, display/face, sound, Wi-Fi controller and OTA. No Raspberry Pi.

The name is a homage to Microduck. MicroCluck is a personal project, not affiliated with Pollen Robotics or Hugging Face.

---

## 2. Goals and non-goals

### v1 goals

| ID | Goal |
|----|------|
| G1 | Stand and walk on flat indoor floors from velocity commands (forward/back, strafe, turn). |
| G2 | Single board: all compute on the Freenove ESP32-S3. |
| G3 | Phone control from any browser (iOS/Android), no app install. |
| G4 | Expressive chicken face on the torso screen, mapped to robot state, with touch interaction (§7.6). |
| G5 | Safe failure: dead-man stop, fall detection, servo protection, low-battery handling, E-stop. |
| G6 | OTA firmware/policy updates and live telemetry for sim2real debugging. |
| G7 | Chicken personality: IMU-stabilized eyes (the body moves, the face holds still) and cluck sound effects. |

### Non-goals (v1)

- Camera / vision, voice interaction, ToF / obstacle sensing, autonomous navigation
- Rough terrain
- Standing up after a fall (stretch goal, v1.1)
- Direct Bluetooth gamepad pairing (ESP32-S3 is BLE-only; physical controllers go through the phone, see §7.5)
- Any commercial use (Microduck meshes are CC BY-NC-SA; see §3)

---

## 3. Reference design and constraints

The design deliberately stays close to Microduck so its training stack, actuator model and leg kinematics can be reused.

| Item | Microduck (reference) | MicroCluck |
|---|---|---|
| Height / mass | 25 cm, 780 g | ~25 cm, **≤ 800 g budget** |
| Servos | 15 (legs, neck/head, beak) | **10** (legs only) |
| Leg joints (per leg) | hip_yaw, hip_roll, hip_pitch, knee, ankle | Same (keep hip_yaw for turning) |
| Servo model | Dynamixel XL330 (BAM actuator model) | **Dynamixel XL330-M288-T** (verify variant, §11) |
| Compute | Rockchip RK3566 + NPU, Linux | ESP32-S3, FreeRTOS |
| Policy | PPO (mjlab), 50 Hz, ONNX, obs 61 → act 14 | Same recipe, retrained: obs ~39 → act 10 |
| Extras | Camera, ToF, mics, NFC, beak | Screen face, speaker, comb, tail, three-toed feet; claw in v2 |

**Facts carried over from Pollen's repos (see References):**

- Policies are trained at 50 Hz with PPO, exported to ONNX; the exporter bakes the observation normalizer into the ONNX graph.
- Actor observation is 61-dim: 48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`.
- Actuator fidelity is the dominant sim2real gap at this scale; the sim models the XL330 down to its voltage control law, with domain randomization on battery voltage, sag, command delay and friction. A **Backlash** variant of each task models ±1° gear play.
- Training needs a CUDA GPU (~1–2 h for a usable gait at 4096 envs on their hardware). Hugging Face Jobs is a fallback.
- **Licensing:** code is Apache-2.0; STL meshes are **CC BY-NC-SA 4.0** (non-commercial); the CAD is not published and Pollen asks that the mechanical design not be called open-source hardware.

**XL330-M288-T key specs (ROBOTIS e-Manual):** 18 g · 0.52 N·m stall @ 5 V (1.47 A) · input 3.7–6.0 V, 5.0 V recommended · TTL half-duplex, **3.3 V logic** (5 V tolerant) · 9.6 kbps–4 Mbps · Protocol 2.0 · feedback includes position, velocity, current, temperature, input voltage. Estimated continuous (rated) torque is only ~0.10 N·m (20% of stall) — thermal management matters (§7.8).

---

## 4. System architecture

```
 Phone browser ── Wi-Fi (SoftAP) / WebSocket ──►  ESP32-S3  (Freenove FNK0104, 3.5")
                                                   │
                                                   ├─ Core 1:  imu_task (500 Hz–1 kHz)
                                                   │           control_task (50 Hz, highest prio)
                                                   ├─ Core 0:  Wi-Fi + HTTP/WS, LVGL face, audio,
                                                   │           telemetry, SD logging, OTA
                                                   │
                                                   ├─ UART1 ─► 74HC126 half-duplex buffer ─► 10× XL330 (daisy chain)
                                                   ├─ I2C (shared with touch) ─► IMU, INA226 power monitor
                                                   └─ I2S ─► onboard codec/amp ─► speaker (verify on 3.5")

 2S Li-ion pack ──┬─► fuse ─► 5 V high-current buck ─► servo bus (+ bulk caps)
                  ├─► 5 V logic buck ─► Freenove 5 V input
                  └─► INA226 (pack V / I)
```

---

## 5. Hardware

### 5.1 Bill of materials (v1, estimates — verify prices before ordering)

| Item | Qty | Notes | Est. cost |
|---|---|---|---|
| Freenove ESP32-S3 Display, 3.5" IPS touch (FNK0104 family) | 1 | Already owned; ships with a speaker | ~$22–30 |
| Dynamixel XL330-M288-T | 10 (+2 spare) | Legs; spares strongly recommended | ~$24–29 ea |
| IMU breakout — ICM-42688-P (I2C) | 1 | Default; alt BNO085 (see §11) | ~$10–20 |
| INA226 power monitor breakout | 1 | Pack V/I over I2C — uses zero GPIO | ~$5 |
| Half-duplex buffer: 74HC126 (DIP-14, through-hole), small perfboard, 10 kΩ pull-up, 0.1 µF decoupling cap | 1 | Through-hole for easy soldering; see §5.2 servo bus interface | ~$5 |
| 2S Li-ion/LiPo pack, ~1500–2500 mAh, with BMS | 1 | + charger | ~$20–35 |
| 5 V buck, servo rail, ≥ 8 A peak | 1 | Low ripple; add ≥ 1000 µF bulk | ~$10–20 |
| 5 V buck, logic, 2 A | 1 | Separate from servo rail | ~$5 |
| Power switch, XT30, inline fuse, JST-EH Dynamixel cables | — | | ~$15 |
| 3D-printed frame, feet, comb, tail, bumpers (PLA/PETG + TPU), M2 hardware | — | | ~$20–30 |
| **Total** | | | **~$400–510** (incl. 2 spare servos) |

### 5.2 Pin map (target)

> ⚠️ Derived from the **2.8" FNK0104B** pinout. The 3.5" variant must be verified against Freenove's pin chart before wiring (Milestone M0, task 1).

| Function | GPIO | Header | Notes |
|---|---|---|---|
| Servo bus UART TX | IO21 | P3 "Expanded IO" | UART1 via GPIO matrix |
| Servo bus UART RX | IO14 | P3 | |
| Servo bus direction (RTS) | IO2 | P3 | Drives the 74HC126 gate enable; ESP-IDF `UART_MODE_RS485_HALF_DUPLEX` toggles it automatically |
| IMU data-ready interrupt | IO3 | P3 | Input only. IO3 is a strapping pin (JTAG source) — harmless unless JTAG eFuse is set; verify |
| I2C SDA / SCL | 16 / 15 | P4 (I2C) | Shared with touch controller; IMU 0x68/0x69, INA226 0x40 — verify touch address on 3.5" |
| Onboard (display, touch, SD, audio codec + amp, BOOT=IO0) | — | — | Used by the board; do not repurpose |

This consumes **all four** free expansion GPIOs. Anything added later (claw servos, sensors) must go on the servo bus or I2C.

**Servo bus interface (74HC126, through-hole)**

```
ESP32 TX  (IO21) ──► 74HC126 gate A input
ESP32 DIR (IO2)  ──► 74HC126 gate A enable (high = transmit)
                     74HC126 gate A output ──┬──► DATA (XL330 TTL bus)
ESP32 RX  (IO14) ◄───────────────────────────┤
                     10 kΩ pull-up to 3.3 V ─┘   (keeps the idle bus high)

74HC126 VCC = 3.3 V with 0.1 µF decoupling. Unused gates: inputs and enables tied to GND.
Bus GND shared with ESP32 GND. Servo 5 V does NOT connect to the buffer.
```

- One tri-state gate drives the bus only while DIR is high; otherwise it releases the line so the servos can reply.
- RX is wired directly to the data line, so it also hears the ESP32's own transmissions. The `dxl` driver must discard that echo (or configure the UART's RS-485 mode to ignore RX during TX).
- 3.3 V logic matches the XL330's TTL level, so no level shifting is needed.
- Built on perfboard with DIP parts and a socket for the chip; no surface-mount soldering in the build.

### 5.3 Power

- **Servo rail:** regulated **5.0 V** (XL330 recommended). Check the voltage domain-randomization range in the sim config and keep the rail inside it.
- **Peak current:** theoretical worst case 10 × 1.47 A ≈ 15 A (all stalled). Size the buck for **≥ 8 A peak**, add bulk capacitance at the bus entry, fuse the pack.
- **Logic rail:** separate 5 V buck into the Freenove board. Common ground at a single star point.
- **Monitoring:** INA226 for pack voltage/current; XL330 *Present Input Voltage* register for the servo rail.

### 5.4 Mechanical

- **Legs:** reuse Microduck leg geometry from the STLs in `microduck_rl` (non-commercial license) to keep kinematic parity with Pollen's model. Only the torso and feet change.
- **Torso:** screen frame with a **raised bezel / rubber corner bumpers**, glass recessed 2–3 mm. Forward falls will land on the screen. Bezel color per renders (yellow).
- **Comb:** snap-on piece on a **top mounting boss**. The v2 claw mounts on the same boss and replaces the comb (or the comb becomes the claw's cover).
- **Tail:** rear feathers in slightly flexible material (TPU) so the tail doubles as a **bumper for backward falls** and a **third contact point when sitting**. It shifts the center of mass rearward, so its mass and collision shape go into the sim model.
- **Feet:** three-toed chicken feet on a **flat sole**; the sole is the designed contact surface. Shorten the inner toe, or include the toes in the sim's collision model, so the policy learns to avoid foot-to-foot clipping on narrow steps.
- **Stance (hip spacing):** decide explicitly. Wider spacing stands steadier but makes each step's side-to-side weight shift larger. Whatever spacing is built goes into the MJCF (§6.1).
- **IMU:** rigidly mounted to the torso/pelvis frame. Record its exact pose relative to the model's base link for the MJCF.
- **Cabling:** daisy-chain routing with strain relief across hip joints.

**Mass budget (estimate, to be replaced with weighed values):**

| Part | Est. mass |
|---|---|
| 10× XL330 | 180 g |
| Freenove 3.5" board | TBD (weigh) |
| Battery (2S) | 70–100 g |
| Frame, feet, bumpers | ~200–250 g |
| Comb + tail | ~15–30 g |
| Electronics, buck converters, cables | ~50–60 g |
| v2 claw reserve | ~60–100 g |
| **Target total (v1)** | **≤ 800 g** |

---

## 6. Simulation and training

### 6.1 Fork and adapt `microduck_rl`

1. Fork `pollen-robotics/microduck_rl` (include as `sim/` submodule).
2. Duplicate the MJCF robot models (`robot_walk.xml`, `robot_groundcontact.xml`, backlash variants) as MicroCluck versions.
3. Remove neck/head joints and bodies (`neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`) and the beak.
4. Add the torso body: box geom(s) matching **measured** mass, CoM and inertia of the screen board + frame + battery + **comb + tail**. Give the tail a collision geom (it touches the ground in backward falls and when sitting).
5. Replace the feet with the MicroCluck sole (and toes, if not shortened). Set the **hip spacing** to the as-built value.
6. Change joint count 14 → 10 everywhere: actuators, observations, actions, joint-index constants, and the repo's regression tests (they lock in joint-index mappings).
7. **Observation contract v1 (proposed, 39-dim):** gyro (3) + projected gravity (3) + joint pos (10) + joint vel (10) + last action (10) + twist command (3). Confirm the composition of Pollen's 48-dim proprio block in `mdp.py` before freezing this.
8. **Actor network:** small MLP, target **256-128-64, ELU** (~52k params, ~208 KB float32). Critic size is unconstrained (not deployed).
9. **Domain randomization:** keep Pollen's defaults, plus:
   - torso mass/CoM ±10–15% (measurement error)
   - **payload mass 0–100 g at the top of the torso** (so v1 walking survives the v2 claw)
10. Train the **Backlash** variants.
11. Train locally on the RTX 3060 (12 GB); reduce `num-envs` if VRAM-limited. HF Jobs as fallback.

**v1 tasks:** `Velocity-Flat` (walk; standing = zero command). **v1.1:** `VelStand` (walk + fall recovery), `SitStand` (sitting can use the tail as a support point).

### 6.2 Policy export pipeline (`tools/policy_export/`)

1. Input: ONNX from Pollen's `scripts/export.py` (normalizer baked in). Never hand-convert a checkpoint.
2. Extract weights, biases, activations and the normalizer. Fold the normalizer into the first layer: `W' = W / σ`, `b' = b − W·(μ/σ)`.
3. Emit `policy_<name>.h`: C arrays + metadata (obs dim, act dim, action scale, default pose, **contract version hash**).
4. Generate golden vectors: ≥ 1000 plausible observations → ONNX outputs.
5. **On-device test:** C inference vs golden, max abs error ≤ 1e-4.
6. **Host closed-loop test:** compile the same C inference for the host and run it in MuJoCo; behavior must match the ONNX policy.

### 6.3 Sim2real contract (`docs/sim2real-contract.md`)

Single source of truth, versioned and hashed. Must match Pollen's runtime (`robotd`) and constants (`microduck_constants.py`) exactly:

- Observation order, units and scaling
- **Joint velocity source** (servo register vs finite difference of position) — match what the sim/runtime assumes
- Action post-processing: action scale, default (HOME) pose offset, clipping
- **Servo control mode and firmware PD gains** (the sim emulates the servo's firmware PD; gains must match)
- Command ranges (clamp phone commands to the training ranges)
- Loop rate (50 Hz) and expected command delay

Firmware refuses to run a policy whose contract hash doesn't match its build.

---

## 7. Firmware

### 7.1 Stack

- **ESP-IDF v5.x**, C (or C++), FreeRTOS
- LVGL via `esp_lvgl_port`; **ESP-DSP** for dot products/matmul
- `esp_http_server` with WebSocket support; static assets in LittleFS
- I2S driver + onboard codec for sound effects
- Native OTA with A/B partitions and rollback
- SD card logging

Rationale over Arduino: explicit core pinning and priorities, RS-485 half-duplex UART mode, `esp_timer` precision, partition control for OTA.

### 7.2 Task layout

| Task | Core | Priority | Rate | Responsibility |
|---|---|---|---|---|
| `control_task` | 1 | Highest | 50 Hz (`esp_timer`) | Snapshot IMU state → Sync Read servos → build obs → infer → post-process → Sync Write. Safety checks. One round-robin servo health read per cycle. |
| `imu_task` | 1 | High | 500 Hz–1 kHz (IMU INT) | Read IMU, gyro bias removal, Mahony filter → gyro + gravity vector (double-buffered) |
| `net_task` | 0 | Medium | Event-driven | Wi-Fi, HTTP, WebSocket commands in, telemetry out |
| `ui_task` | 0 | Low | ≤ 30 fps | LVGL face rig, expressions, eye stabilization, status strip, touch |
| `audio_task` | 0 | Low | Event-driven | Sound effect playback via I2S DMA |
| `log_task` | 0 | Low | Buffered | Obs/actions/timing to SD (binary or CSV) |
| OTA | 0 | Low | On demand | Firmware + policy updates |

Shared state between cores via double buffers or lock-free single-writer structures. No blocking calls in `control_task`.

### 7.3 Timing budget (per 20 ms cycle)

| Step | Budget |
|---|---|
| Sync Read 10 servos (pos + vel), 2 Mbps, Return Delay Time = 0 | ~1–2 ms |
| Observation build | < 0.2 ms |
| Policy inference | **≤ 5 ms target** |
| Sync Write 10 goal positions | < 0.5 ms |
| Slack | ≥ 10 ms |

**Requirements:** loop period p99 jitter ≤ 1 ms **with Wi-Fi, UI (including eye stabilization) and audio active**; cycle-overrun counter exposed in telemetry. Set each servo's Return Delay Time to 0 (default adds delay per servo). Evaluate Fast Sync Read if supported by servo firmware.

### 7.4 Memory plan

- **Internal SRAM:** Wi-Fi/LwIP, task stacks, DMA-capable display partial buffers, I2S DMA buffers, bus/IMU buffers.
- **PSRAM:** LVGL non-DMA buffers, face accessories (heart, "?", Zzz), sound clips (or stream from LittleFS/SD), policy weights (one ~208 KB set per behavior; several fit), log ring buffer.
- Measure inference time with weights in PSRAM first. Move the active policy to internal SRAM or quantize to int8 **only if** > 5 ms.
- Track heap high-water marks in telemetry.

### 7.5 Phone controller and protocol

**Networking**
- Default: SoftAP, SSID `microcluck-XXXX` (WPA2), robot at `192.168.4.1`.
- Optional STA mode (join home Wi-Fi) for development, with mDNS `microcluck.local`.

**Pages**
- `/` — controller: static HTML/JS from flash. Virtual joystick (vx, vy), turn control (wz), behavior buttons, expression buttons, **E-stop**. Reads the browser **Gamepad API** so any controller paired to the phone (PS5, Xbox) works.
- `/status`, `/settings` — server-rendered fragments (HTMX-style) for config, mode, Wi-Fi, gains, volume/mute.
- `/ota` — firmware/policy upload.

**WebSocket `/ws`** (compact JSON v1; binary later if needed)

| Direction | Message | Rate |
|---|---|---|
| client → robot | `{"t":"cmd","seq":123,"vx":0.10,"vy":0.0,"wz":0.30}` | 20–30 Hz |
| client → robot | `{"t":"act","name":"sit"}` / `{"t":"estop"}` / `{"t":"face","expr":"love"}` / `{"t":"sound","name":"cluck"}` | On event |
| robot → client | `{"t":"tel","mode":"WALK","expr":"happy","vbat":7.6,"loop_p99_ms":0.6,"overruns":0,"tilt_deg":4.2,"servo_tmax_c":48,"rssi":-52}` | 10 Hz |

**Dead-man:** no `cmd` for **300 ms** → velocity command = 0 (stand). No connection for 5 s → stand (optionally sit).

### 7.6 Face, sound and UI

**Visual style:** flat vector chicken face on a black background: large white eyes with dark pupils, small arched brows, orange beak, pink cheeks. The **beak is the fixed anchor**; the eyes, brows and accessories do the emoting.

**Layout (portrait 320×480):** face occupies the upper two-thirds. A slim **status strip** in the lower third shows battery, Wi-Fi and mode. Accessories (heart, "!", "?", Zzz) pop up in the space above and beside the eyes.

**Expression set and state mapping**

| Expression | Trigger | Look |
|---|---|---|
| Curious (default) | IDLE; STAND with no command | Open eyes, idle blinks, slow gaze drift |
| Happy | WALK; trick completed | Eyes arc closed, beak open; joy marks angled upward above the eyes (not sideways — side lines read as cat whiskers) |
| Surprised | Bump / near-fall (gyro or tilt spike below the fall threshold) | Mismatched eye sizes, spark marks |
| Love | Screen tap ("petting") or phone button | Eyes closed in arcs, heart, cheeks emphasized |
| Dizzy | FALLEN | Spiral eyes |
| Confused | Dead-man triggered / phone disconnected | Tilted brow and "?" |
| Sleepy | Low-battery warning; long idle | Half-closed lids, Zzz |
| Error | Fault or ESTOP | Visually distinct from every mood (e.g., X eyes + red status strip) so it is never mistaken for one |

The renders show the first four (curious, happy, surprised, love); the rest are to be designed in the same style.

**Parametric face rig (implementation requirement):** the face is built from LVGL primitives (circles, arcs, triangles), not stored bitmaps. Parameters:

- per eye: openness (0–1), scale, pupil offset (x, y), lid curve (open ↔ happy arc)
- brow angle per side
- beak open (0–1)
- cheek alpha
- accessory (none / heart / spark / "?" / Zzz / spiral)

Each expression is a **preset** of these parameters. Transitions tween over ~150–300 ms. Blinks fire every 3–6 s at random. Gaze follows the commanded turn (`wz`) and strafe (`vy`). Small accessory icons may be images in PSRAM.

**IMU-stabilized eyes (signature feature):** like a real chicken's head, the face holds still while the body moves. The whole face group is offset opposite to torso roll (→ x) and pitch (→ y), using the gravity vector from `imu_task`, low-pass filtered and clamped to about ±20 px. Disabled in FALLEN and ESTOP. Redraw only the face bounding box, at ≤ 30 fps, and verify core-0 load stays acceptable (§10).

**Sound:** onboard codec + speaker (verify on the 3.5" variant). Clips stored in LittleFS or on SD, played by `audio_task`:

- Cluck when starting to walk and occasionally while walking (rate-limited, not every step)
- Startled squawk on near-fall / fall
- Happy cluck on "petting"
- Sleepy coo on low battery / long idle

Volume and mute in `/settings`. Sound playback must never affect control-loop timing.

**Touch:** tap = love/react; **long-press = E-stop (torque off)**.

### 7.7 Mode state machine

```
BOOT → IDLE (torque off)
IDLE --enable--> STAND_INIT (interpolate to HOME pose over ~2 s, low gains)
STAND_INIT → STAND (policy, zero command)  ⇄  WALK (policy, phone command)
STAND/WALK --tilt > 60° for > 200 ms--> FALLEN (torque off or compliant) → IDLE
ANY --E-stop--> ESTOP (torque off; requires explicit re-enable)
ANY --low battery--> controlled stop → IDLE
(v1.1) FALLEN --recover--> RECOVER (stand-up policy) → STAND
```

Each mode selects its default expression per §7.6; event-driven expressions (surprised, love) temporarily override it.

### 7.8 Safety requirements

| ID | Requirement |
|---|---|
| S1 | E-stop from phone and from screen long-press disables torque within one control cycle. |
| S2 | Dead-man: 300 ms without commands → zero velocity. |
| S3 | Fall detection per §7.7; torque off/compliant to protect gears and screen. |
| S4 | Servo health: round-robin read of temperature and hardware error status; on overheat threshold → controlled sit/stand-still; on hardware error → torque off and report. |
| S5 | Low battery (INA226): warn, then controlled stop and torque off before BMS cutoff. |
| S6 | Command clamping to the policy's training ranges. |
| S7 | Policy contract hash mismatch → refuse to enable torque. |
| S8 | Host-side unit tests for S1–S7 logic. |

---

## 8. Milestones (incremental; each must meet exit criteria before the next starts)

| # | Milestone | Exit criteria |
|---|---|---|
| **M0** | Bench bring-up (no robot) | **Verify 3.5" pinout, touch address, PSRAM size, audio codec + speaker.** Parametric face rig with the curious preset and blinks; SoftAP + controller page + WebSocket echo; OTA round-trip with rollback; SD logging; one sound clip plays. |
| **M1** | Servo bus | Half-duplex circuit built; 10 servos ID'd, 2 Mbps, Return Delay 0; 50 Hz Sync Read/Write for 10 min with **zero comm errors** and **p99 jitter ≤ 1 ms with Wi-Fi + UI + audio active**; torque enable/disable; health polling. |
| **M2** | IMU | Gyro bias calibration at boot; gravity vector within ±2° on known tilts; stable at 500 Hz+. Eye stabilization demo on the bench (tilt the board, eyes stay put). |
| **M3** | Sim model + policy port | Forked repo with 10-DOF MicroCluck model using measured torso (incl. comb + tail), as-built feet and hip spacing; walking policy trained (backlash variant); export pipeline; **golden test passes on device**; host closed-loop test matches ONNX. |
| **M4** | First steps | Suspended (gantry/string) test → assisted standing → free standing → walking on flat floor. Telemetry and logs captured and compared to sim. |
| **M5** | v1 complete | Phone control polished (incl. Gamepad API); full expression set mapped to states; stabilized eyes while walking; sounds; S1–S8 verified on hardware. |
| M6 | v1.1 | Fall recovery (`VelStand`) and sit/stand policies (tail-assisted sit); policy hot-swap. |
| — | v2 | Claw (§9). |

---

## 9. v2 claw (future scope)

- Claw servos join the **same Dynamixel bus** — no new GPIO.
- Mounts on the **comb's top boss**; the comb is removed or reworked as the claw's cover.
- **Option A:** fixed gripper on top of the torso for carrying hand-offs (simplest).
- **Option B:** 1–2 DOF arm (shoulder pitch + gripper) that swings forward/down; floor pickup requires a crouch-and-reach policy (cf. Microduck's `GroundPick` task, which uses the beak).
- v1 prepares for it: shared mounting boss, mass reservation, payload randomization in training (§6.1).
- Keep total mass ≤ ~850 g; XL330s are sized for Microduck's ~800 g.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sim2real gap | Same servos as Pollen's actuator model; backlash variant; measured torso properties (incl. comb + tail); sim2real contract; heavy logging. |
| 3.5" pinout differs from 2.8" | Verify first (M0). Fallback: re-map UART/INT via GPIO matrix; worst case move IMU INT to polling. |
| Wi-Fi interrupts cause loop jitter | Core isolation; measure in M1 with Wi-Fi + UI + audio under load. |
| Stabilized face forces large redraws every frame | Face-bounding-box dirty area, ≤ 30 fps, clamp offset; measure core-0 load; reduce fps or disable stabilization while walking fast if needed. |
| Internal RAM pressure (Wi-Fi + LVGL + I2S + policy) | PSRAM placement; small actor; heap high-water telemetry. |
| Servo overheating (rated torque only ~20% of stall) | Temperature monitoring; auto-rest; keep mass low; avoid long deep-crouch stands. |
| Screen damage in falls | Raised bezel, bumpers, recessed glass; tail as rear bumper. |
| Toes clip the opposite foot | Shorten inner toes or include toes in sim collisions. |
| Brownouts from servo spikes + Wi-Fi TX + audio | Separate rails, bulk caps, BMS sized for peaks. |
| Non-commercial mesh license | Personal project only; redesign legs before any commercial use. |
| Cost/availability of XL330 | Order spares early; servos are most of the BOM. |

---

## 11. Open questions

1. 3.5" FNK0104 exact pinout, touch controller + I2C address, PSRAM size, 5 V input pin location, and whether it has the same audio codec + amp as the 2.8".
2. Which XL330 variant (M288 vs M077) Pollen's actuator model targets — check `microduck_constants.py` before ordering.
3. Exact composition of Pollen's 48-dim proprio block and action post-processing (`mdp.py`, `robotd`).
4. IMU: ICM-42688-P + on-MCU Mahony (default) vs BNO085 on-chip fusion (I2C quirks on ESP32 are a known concern).
5. Battery: 2S pack (default) vs alternatives; target runtime.
6. Leg geometry: reuse Microduck STLs as-is vs redesign.
7. Hip spacing (stance width) for the as-built torso.
8. Firmware language: C (default) vs C++ vs Rust (esp-rs).
9. Designs for the remaining expressions (dizzy, confused, sleepy, error).

---

## 12. Proposed repo layout

```
microcluck/
├── CLAUDE.md                   # Working notes for Claude Code (see §13)
├── docs/
│   ├── PRD.md                  # This document
│   ├── sim2real-contract.md    # Obs/action contract (versioned, hashed)
│   ├── face-spec.md            # Expression presets, rig parameters, state mapping
│   ├── wiring.md               # Pin map, power, schematics
│   ├── bom.md
│   └── decisions/              # ADRs (one file per decision)
├── firmware/                   # ESP-IDF project
│   ├── main/
│   └── components/
│       ├── dxl/                # Dynamixel Protocol 2.0, sync read/write, health
│       ├── imu/                # Driver + Mahony filter
│       ├── policy/             # C inference + generated headers
│       ├── control/            # Control loop + state machine
│       ├── safety/             # Dead-man, fall, thermal, battery, E-stop
│       ├── net/                # SoftAP/STA, HTTP, WebSocket, OTA
│       ├── ui/                 # LVGL face rig, expressions, stabilization, status strip
│       ├── audio/              # I2S codec, sound effects
│       └── log/                # SD logging
├── web/                        # Controller page (bundled into firmware LittleFS)
├── assets/                     # Sound clips, face accessory icons
├── tools/
│   └── policy_export/          # ONNX → C header, golden vectors, host tests
├── sim/                        # git submodule: fork of microduck_rl
└── hardware/                   # CAD, STLs (legs, feet, comb, tail), wiring diagrams
```

---

## 13. Working notes for Claude Code

- Work milestone by milestone (§8). Don't start a milestone until the previous exit criteria are met.
- `docs/sim2real-contract.md` is the source of truth for observation/action layout. Firmware and sim must reference the same contract hash.
- `docs/face-spec.md` is the source of truth for expression presets and the state → expression mapping.
- Never hand-edit generated policy headers; regenerate via `tools/policy_export`.
- Nothing blocking or allocating inside `control_task`. UI and audio must never affect its timing.
- Safety logic (§7.8) must have host-side unit tests.
- Hardware-in-the-loop steps (wiring, flashing, gantry tests, weighing parts) are manual — flag them for Neil rather than assuming results.
- Record significant decisions as ADRs in `docs/decisions/`.

---

## 14. References

- Pollen Robotics — Microduck product page: https://store.pollen-robotics.com/products/microduck
- `pollen-robotics/microduck` (onboard runtime): https://github.com/pollen-robotics/microduck
- `pollen-robotics/microduck_rl` (training, MJCF, export): https://github.com/pollen-robotics/microduck_rl
- ROBOTIS XL330-M288-T e-Manual: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- Freenove FNK0104 (ESP32-S3 Display): https://store.freenove.com/products/fnk0104 · tutorial: http://freenove.com/fnk0104
- FNK0104B (2.8") pinout reference: https://github.com/MrTartuf0/onda-flac-player/wiki/Hardware
- Open Duck Mini v2 (predecessor project): https://github.com/apirrone/Open_Duck_Mini
