# Microduck 1v1 football

Football uses a dedicated two-robot MuJoCo scene and `football-train` runner. One policy controls each robot. The field is 2.0 × 1.2 m with physical sidelines and 35 cm goal openings. Balls and the original duck collision meshes contact the walls directly. A duck whose root escapes the field still forfeits. Control runs at 50 Hz; each match lasts up to 1,500 steps (30 s). Goals reset the ball and players at center while the score and match clock continue. A duck whose root leaves the 2.0 × 1.2 m field forfeits immediately: it gets a −15 terminal reward, its opponent gets +15 and wins regardless of the goal score. Simultaneous exits are a draw. Each new match randomizes the attacking directions and learner seat.

## Policy contracts

The tactical actor observes 64 values and emits `(vx, vy, yaw rate, kick request)`. The first 48 values match the existing Microduck proprioception order; the remaining 16 are noisy, one-step-delayed relative ball and opponent motion, attack and own goal directions, opponent heading, kick availability, and time remaining. A privileged 82D critic sees exact state. The tactical output passes through frozen walking, right-foot kick, and recovery policies to produce 14 raw joint-position offsets. The joint actor observes the same 64 values and emits 14 offsets directly; it needs no skill controller for inference. Its observation contract is separate from the existing 61D runtime policies.

## Train

Use compatible RSL checkpoints for walking, right-foot kick, and recovery. All three actors must accept 61D observations and emit 14D joint offsets. Their MLP width must match the football joint actor `(512, 256, 128)` for the walking warm start. The tactical run copies these exact files into `skills/` and records SHA-256 hashes, so later stages need only the preceding football checkpoint.

```bash
uv run football-train tactics \
  --walk-checkpoint policies/football/walk.pt \
  --kick-checkpoint policies/football/kick.pt \
  --recovery-checkpoint policies/football/recovery.pt \
  --num-envs 1024 --iterations 6000 --seed 1 --run-name seed-1

uv run football-train distill \
  --teacher-checkpoint logs/football/tactics/seed-1/model_best.pt \
  --num-envs 1024 --iterations 2000 --seed 1 --run-name seed-1

uv run football-train joint \
  --init-checkpoint logs/football/distill/seed-1/model_best.pt \
  --num-envs 1024 --iterations 6000 --seed 1 --run-name seed-1
```

For the required short integration checks, substitute `--num-envs 64 --iterations 5` at each stage. To continue an interrupted stage, use `football-train <stage> --resume logs/football/<stage>/<run>/model_<n>.pt --num-envs <same count> --iterations <additional iterations>`. A resume restores actor, critic, optimizer, normalizers, skill hashes, opponent snapshots, pool RNG, and training iteration. It starts a fresh match because physical simulator state is not stored.
Existing football checkpoints remain loadable with the updated field and kick behavior. A resumed run starts using the new rules from its next match; re-evaluate an earlier tactical checkpoint before choosing it as a teacher.

A nonfinite simulator world or joint action voids that match: only that world resets, both players receive zero reward on the failed transition, and the match is excluded from match-point statistics. The `/football/numerical_resets` log entry counts these resets. To investigate a numerical failure, add `--debug-numerics` to an isolated resume. It checks joint actions and simulator state at the physics boundary, raises on the first nonfinite value, and saves the preceding 20 physics states plus the MuJoCo model under `<run>/nan_dumps/`. The exception reports affected environment IDs, match elapsed time, and prior out-of-bounds forfeits. This diagnostic aborts before automatic recovery; it slows training and should be used for investigation.

Training records a 6-second view of vector environment 0 at startup and every 200 PPO iterations by default. Clips are saved under `logs/football/<stage>/<run>/videos/train/`; recording resumes on the global iteration schedule after a checkpoint load. Each MP4 has a same-stem `.jsonl` file with one row per captured frame. Rows include the ball and both ducks' world positions, orientations, velocities, and out-of-bounds flags; learner seat and opponent identity; home/away joint actions; kick timers and the scripted failed-kick and near-ball-stall flags; scores before goal and timeout bookkeeping; and global control step. Positions and velocities are sampled after each physics step, aligned with the corresponding frame. Check `home.position_w` and `away.position_w` against `env_origin_w` to find a duck outside the camera view. The JSONL stays local.

The learner keeps the robot's original white and orange colors in training, evaluation, and the playground. Opponents are gold for scripted, blue for frozen tactical, and purple for frozen joint policies. Colors follow the assigned learner seat at each match reset; the training recorder shows environment 0.

Use `--video-every N` to change the interval, `--video-steps N` to change clip length (50 frames per second), or `--video-every 0` to disable recording. Clip length cannot exceed the interval (`N × 24` frames); a run that ends sooner records a shorter final clip. Resumed runs use distinct filenames. With `--logger wandb`, completed MP4s are also uploaded to the W&B run in the `mjlab_microduck` project; checkpoints and videos remain local too. For example:

```bash
uv run football-train joint --init-checkpoint logs/football/distill/seed-1/model_best.pt \
  --num-envs 1024 --iterations 6000 --logger wandb \
  --video-every 200 --video-steps 300
```

Distillation uses **student-generated PPO trajectories**. At each pre-action student observation, a frozen tactical teacher and its skill adapter supply a 14D raw action target. The student takes the actual environment action; PPO optimizes match reward and a Huber loss against the target. The guidance weight decays linearly from 1.0 to 0.1 over 2,000 updates. The first 48 input columns and compatible downstream layers come from the walking actor, the new 16 columns start at zero, and the critic starts afresh. Joint self-play removes the guidance loss.

The bundled `policies/football/kick.pt` was validated in its native BallKick task: the ball moved after 0.12 seconds and reached 1.46 m/s. It also kicked from a stationary pose in the football scene. Misses during a moving scripted approach are therefore a skill-transition issue; a failed attempt should not be taken as evidence that the checkpoint cannot kick.

The scripted chase-and-kick opponent approaches the ball at the right-foot kick skill's trained placement (about 9 cm forward and 4.2 cm right), waits until its root speed is below 0.06 m/s, then requests a kick. If the ball has moved less than 1.5 cm after one second of the kick sequence, it cancels that attempt and walks through the ball for the rest of that kickoff. The same fallback activates if the duck and ball remain nearly stationary within 20 cm of each other for one second without a kick. A goal or new match clears this fallback. The controller still rejects kicks outside the placement window. Tactical opponents are scripted until the first snapshot. Every 200 updates, a frozen policy enters the pool; the latest eight are retained. Joint self-play draws 25% tactical and 75% joint opponents once joint snapshots exist, choosing one frozen opponent per match. `model_best.pt` is based on recent completed training match points; compare candidates with the fixed-seed evaluator before final selection.

## Evaluate

Use `football-eval` to play 400 fixed-seed, side-balanced matches against the scripted chase-and-kick baseline or a frozen football checkpoint. For example:

```bash
uv run football-eval --checkpoint logs/football/joint/seed-1/model_best.pt \
  --matches 400 --num-envs 64 --seed 100 --json logs/football/joint/seed-1/eval-scripted.json \
  --video logs/football/joint/seed-1/eval-video
uv run football-eval --checkpoint logs/football/joint/seed-1/model_best.pt \
  --opponent-checkpoint logs/football/tactics/seed-1/model_best.pt \
  --matches 400 --num-envs 64 --seed 100
```

For a useful comparison, run the same seed and match count for tactical, distilled, and joint checkpoints, and compare against prior tactical and joint snapshots. Targets are ≥60% of available baseline match points for the tactical teacher, a distilled student within 10 percentage points, and a joint player at least as strong as the teacher. These are training targets, not claimed results.

Evaluation reports learner and opponent boundary forfeits as well as goals and match points. Video inspection is available from the evaluator's optional recording flag. A genuine benchmark requires trained skill and football checkpoints; the synthetic-checkpoint smoke test only validates the pipeline.

## Play against a learned policy

Launch the native MuJoCo viewer while controlling one Microduck through the same tactical walking and right-foot kick adapter used during training:

```bash
uv run football-play \
  --opponent-checkpoint logs/football/joint/seed-1/model_best.pt \
  --side home
```

A tactical, distilled, or joint football checkpoint can be the opponent. Tactical opponents use their frozen walking, right-foot kick, and recovery skills; distilled and joint opponents act directly in 14-joint space. Its run directory must contain the copied `skills/` bundle. The human controls one robot; the learned checkpoint controls the other. The selected opponent and `--side home|away|random` setting persist across matches. Attack direction still changes between matches and is printed in the terminal.

Keep the launching **terminal focused** for controls while watching the viewer window. Arrow Up/Down increase/decrease forward velocity; Left/Right strafe; A/E turn left/right. Commands persist until changed. Space stops movement and cancels a queued kick. L queues one right-foot kick for up to two seconds; it starts when the ball enters the kick zone and then runs its three-second skill sequence. Q quits. The viewer can be closed to exit. The terminal must support single-key input (TTY).

For a noninteractive pipeline check, use `--headless --steps 24`. This sends a zero tactical command and does not open a viewer.
