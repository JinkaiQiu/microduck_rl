# Microduck 1v1 football

Football uses a dedicated two-robot MuJoCo scene and `football-train` runner. One policy controls each robot. The field is a rounded rectangle with independently randomized 2.0–4.4 m length and 1.3–2.2 m width (four straight sides and 0.4 m corner radii) and 35 cm goal openings. Balls and the original duck collision meshes contact the walls directly. A duck forfeits only when its root crosses outward through either goal opening. The pitch has a level center covering most of its area; only the shallow rim near the walls rises toward the boundary. Each match samples the rim height independently and scales the rim with its randomized arena size, so the ball tends to roll back from the edges while central play stays level. Control runs at 50 Hz. The first goal ends the match immediately; the next match starts from fresh randomized player and ball spawns. A scoreless match ends in a draw after 30 seconds (1,500 control steps). A duck whose root crosses outward through a goal opening forfeits immediately: it gets a −20 terminal reward, its opponent gets no boundary-forfeit reward but wins regardless of the goal score. Simultaneous exits are a draw. Each new match randomizes the attacking directions and learner seat. Kickoff places the ducks in opposite own-goal halves, with x between 0.20 and 0.70 m from midfield and y within ±0.28 m. The ball starts in a central band whose x/y range shrinks with smaller fields (at most 25% of half-length and 50% of half-width); it must start at least 0.18 m from either duck root and 0.06 m inside the rounded boundary.

## Policy contracts

The tactical actor observes 66 values and emits `(vx, vy, yaw rate, kick request)`. The first 48 values match the existing Microduck proprioception order; the remaining 18 contain noisy, one-step-delayed relative ball and opponent motion, attack and own goal directions, opponent heading, kick availability, elapsed time capped at the 30-second match deadline, and the exact normalized field half-length and half-width. The arena half-length and half-width are resampled independently per match from 1.0–2.2 m and 0.65–1.1 m; walls, goals, rules, and video geometry follow the samples. The minimum half-width is 0.65 m so the fixed 0.4 m corner radius and 0.175 m half-goal opening leave physical side-wall segments. A privileged 84D critic sees exact state. The tactical actor uses a scaled tanh Gaussian: sampled and deterministic commands are bounded to `|vx| ≤ 0.4 m/s`, `|vy| ≤ 0.3 m/s`, `|yaw rate| ≤ 1 rad/s`, and `|kick request| ≤ 1`. PPO stores these same controller-unit actions and their transformed log probabilities, so the skill adapter does not clip ordinary policy output. The tactical output passes through frozen walking, right-foot kick, and recovery policies to produce 14 raw joint-position offsets. The joint actor observes the same 66 values and emits 14 offsets directly; it needs no skill controller for inference. Its observation contract is separate from the existing 61D runtime policies.


## Ball contact randomization

At each new match, each world independently samples the ball contact damping ratio from 0.4–1.0 and ball–floor sliding friction from 0.75–1.25. The 0.4 lower bound was checked with 1–5 m ball drops; smaller damping made MuJoCo Warp floor impacts add energy and launch the ball. MuJoCo models rebound through contact damping rather than a direct restitution coefficient: lower damping gives a livelier ball. The ball–floor pair has its own friction, so this randomization does not change ball–foot or ball–wall friction. A goal ends the match, so the next match reset samples new contact values. These are absolute samples, so repeated resets do not accumulate changes.

## Rewards and logging

The learner and opponent receive symmetric, signed rewards except for boundary forfeits. Each 50 Hz step pays `ball_progress` for moving the ball toward the attacking goal (2 × forward progress, clipped to ±0.05 m before scaling) and `ball_approach` for moving the right-foot strike point toward the ball's pre-action position (0.5 reward per meter closer at perfect alignment). The strike point is 9 cm forward and 4.2 cm to the right of the duck root, matching the blind kick skill's placement. Positive progress is weighted linearly from full credit when facing the ball to zero at 45° of heading error; beyond ±45° it earns exactly zero, while retreat remains fully negative. Facing and ball position are measured before the action, so a stationary turn or a rolling ball cannot pay approach reward. A `successful_kick` pays +0.5 when a foot–ball contact increases the ball's goalward speed by at least 0.2 m/s and leaves it moving goalward at least 0.25 m/s; a 20-step cooldown prevents repeated payment for one strike. A low body receives `fall_penalty` of −0.02 per step. `boundary_proximity` is −0.005 × exp(−d / 0.05 m) per step, where d is the duck root's inward clearance from the rounded edge in meters. It is about −0.0018 at 5 cm from the edge and exactly zero beyond 10 cm. `body_body_proximity` is a symmetric penalty of −0.001 × exp(−max(root distance − 0.24 m, 0) / 0.05 m), exactly zero when the estimated footprint gap exceeds 10 cm. It uses a 12 cm circular footprint around each duck root; estimated footprint overlap gets the full penalty, while a 5 cm gap gets about −0.00037 per duck per step. The first goal pays ±20 and terminates the match. There is no separate match-outcome reward or time-limit reward. A goal-door forfeit gives only the exiting duck −20, while its opponent receives 0; it clears all other components. Numerical resets clear every component.

TensorBoard and W&B log the learner's mean **weighted reward per control step** under `/football/reward/ball_progress`, `/football/reward/ball_approach`, `/football/reward/successful_kick`, `/football/reward/fall_penalty`, `/football/reward/boundary_proximity`, `/football/reward/body_body_proximity`, `/football/reward/goal`, `/football/reward/boundary_forfeit`, and `/football/reward/total`. The components sum to the total on every transition. Match-point statistics remain separate from reward logging.

`info/benchmark_win_rate` is the raw win fraction over the latest 400 completed matches against scripted or frozen opponents; `info/win_rate_vs_scripted`, `info/win_rate_vs_tactical`, and `info/win_rate_vs_joint` split the same calculation by opponent type. `info/win_rate_vs_live` tracks the first seat in live self-play as a symmetry diagnostic and is excluded from the benchmark rate. Draws count as non-wins, while `info/match_points` still gives a draw half a point. `info/benchmark_matches` and `info/matches_vs_<type>` show each rolling window's sample count; a zero count means its displayed rate has no data yet. Numerical resets are excluded, and boundary forfeits count as completed results. The windows are saved in new checkpoints and restored on resume; older checkpoints begin with empty windows. RSL averages these per-step rolling values over each 24-step PPO iteration, so the logged scalar is a short time average of the current window rather than an exact end-of-iteration snapshot. Use `football-eval` for a fixed-seed win rate.

## Train

Use compatible RSL checkpoints for walking, right-foot kick, and recovery. All three actors must accept 61D observations and emit 14D joint offsets. Their MLP width must match the football joint actor `(512, 256, 128)` for the walking warm start.

`football-train tactics` runs two training phases in one invocation. Its first `--solo-iterations` PPO updates use a physical world with one Microduck, one ball, and the randomized walled arena; there is no second robot or opponent controller. The goal or a 30-second timeout ends each solo episode. The actor uses the same 66D tactical observation and 4D action as later self-play. Its absent-opponent feature slots are exactly zero, and the critic has zero opponent-velocity slots. The command then transfers the actor, critic, and observation normalizers into two-duck tactical self-play, with a fresh optimizer, opponent pool, and match clock. `--iterations` counts the subsequent self-play updates; `--solo-iterations` defaults to 1000.

```bash
uv run football-train tactics \
  --walk-checkpoint policies/football/walk.pt \
  --kick-checkpoint policies/football/kick.pt \
  --recovery-checkpoint policies/football/recovery.pt \
  --num-envs 1024 --solo-iterations 1000 --iterations 6000 \
  --seed 1 --run-name seed-1

uv run football-train distill \
  --teacher-checkpoint logs/football/tactics/seed-1/model_best.pt \
  --num-envs 1024 --iterations 2000 --seed 1 --run-name seed-1

uv run football-train joint \
  --init-checkpoint logs/football/distill/seed-1/model_best.pt \
  --num-envs 1024 --iterations 6000 --seed 1 --run-name seed-1
```

For the required short integration checks, use `--num-envs 64 --solo-iterations 5 --iterations 5` for tactics and `--num-envs 64 --iterations 5` for each later stage. During the tactics warmup, `--num-envs` counts one-duck worlds; during self-play and later stages it counts physical two-duck matches. Two-duck training exposes twice as many PPO observation rows, one per seat; only rows whose actions were executed by the live policy enter PPO updates. To continue an interrupted self-play stage, use `football-train <stage> --resume logs/football/<stage>/<run>/model_<n>.pt --num-envs <same count> --iterations <additional iterations>`. A tactics self-play resume skips the solo warmup. If the warmup itself is interrupted, resume its `solo/model_<n>.pt` checkpoint with `--solo-iterations <additional updates>`; the command then enters self-play. A resume restores actor, critic, optimizer, normalizers, skill hashes, opponent snapshots, pool RNG, and training iteration. It starts a fresh match because physical simulator state is not stored.

Existing 64D/82D football checkpoints remain loadable: training expands actor, critic, normalizer, and optimizer inputs with zero weights for the two new features; frozen opponents and evaluation keep their original input width. A resumed run starts using the new rules from its next match; re-evaluate an earlier tactical checkpoint before choosing it as a teacher.

The physical arena has four straight sides joined by quarter-circle corners. The forfeit check detects outward crossings through the goal openings; positions beyond a solid wall or rounded corner do not forfeit. Existing checkpoints load, but should be re-evaluated under the one-goal match rule.

A nonfinite simulator world or joint action voids that match: only that world resets, both players receive zero reward on the failed transition, and the match is excluded from match-point statistics. The `/football/numerical_resets` log entry counts these resets. To investigate a numerical failure, add `--debug-numerics` to an isolated resume. It checks joint actions and simulator state at the physics boundary, raises on the first nonfinite value, and saves the preceding 20 physics states plus the MuJoCo model under `<run>/nan_dumps/`. The exception reports affected environment IDs, match elapsed time, and prior out-of-bounds forfeits. This diagnostic aborts before automatic recovery; it slows training and should be used for investigation.

Training records a 6-second bird's-eye view of vector environment 0 at startup and every 200 PPO iterations by default. The camera looks straight down and frames the full maximum 4.4 × 2.2 m arena, including its rounded corners, straight walls, and goal openings. Self-play clips are saved under `logs/football/<stage>/<run>/videos/train/`; recording resumes on the global iteration schedule after a checkpoint load. Each MP4 has a same-stem `.jsonl` file with one row per captured frame. Tactics warmup traces under `videos/solo/` contain the ball and one duck; its self-play traces under `videos/train/` and later-stage traces include the ball and both ducks. Rows include the sampled field half-length and half-width and the ducks' world positions, orientations, velocities, and out-of-bounds flags; learner seat and opponent identity; home/away joint actions; kick timers and the scripted failed-kick and near-ball-stall flags; scores before goal bookkeeping; and global control step. Positions and velocities are sampled after each physics step, aligned with the corresponding frame. The trace's `home_out_of_bounds` and `away_out_of_bounds` flags still describe positions outside the rounded pitch; only outward crossings through the goal openings cause a forfeit. Check `home.position_w` and `away.position_w` against `env_origin_w` to find a duck outside the camera view. The JSONL stays local.

A goal, forfeit, or 30-second draw produces a terminal `done` for both policy seats. The wrapper records the result, resets the entire world and skill timers, then returns observations from the new match. PPO masks both its value bootstrap and GAE recursion with that `done`, so future rewards from the new match cannot enter the previous match's returns. A timed-out draw has no extra outcome reward and is not sent to PPO as a bootstrap timeout.

The learner keeps the robot's original white and orange colors in training, evaluation, and the playground. Both ducks keep the original colors when they use the live policy. Opponents are gold for scripted, blue for frozen tactical, and purple for frozen joint policies. Colors follow the assigned learner seat at each match reset; the training recorder shows environment 0.

Use `--video-every N` to change the interval, `--video-steps N` to change clip length (50 frames per second), or `--video-every 0` to disable recording. Clip length cannot exceed the interval (`N × 24` frames); a run that ends sooner records a shorter final clip. Resumed runs use distinct filenames. With `--logger wandb`, completed MP4s are also uploaded to the W&B run in the `mjlab_microduck` project; checkpoints and videos remain local too. For example:

```bash
uv run football-train joint --init-checkpoint logs/football/distill/seed-1/model_best.pt \
  --num-envs 1024 --iterations 6000 --logger wandb \
  --video-every 200 --video-steps 300
```

Distillation uses **student-generated PPO trajectories**. At each pre-action student observation, a frozen tactical teacher and its skill adapter supply a 14D raw action target. In live self-play matches, both student seats take actions and supply PPO transitions and teacher targets. Against a frozen teacher opponent, only the student's executed seat trains. PPO optimizes match reward and a Huber loss against the target. The guidance weight decays linearly from 1.0 to 0.1 over 2,000 updates. The first 48 input columns and compatible downstream layers come from the walking actor, the new 18 columns start at zero, and the critic starts afresh. Joint self-play removes the guidance loss.

The bundled `policies/football/kick.pt` was validated in its native BallKick task: the ball moved after 0.12 seconds and reached 1.46 m/s. It also kicked from a stationary pose in the football scene. Misses during a moving scripted approach are therefore a skill-transition issue; a failed attempt should not be taken as evidence that the checkpoint cannot kick.

The scripted chase-and-kick opponent approaches the ball at the right-foot kick skill's trained placement (about 9 cm forward and 4.2 cm right), waits until its root speed is below 0.06 m/s, then requests a kick. If the ball has moved less than 1.5 cm by the end of the 0.6-second kick sequence, it cancels that attempt and walks through the ball for the rest of that kickoff. The same fallback activates if the duck and ball remain nearly stationary within 20 cm of each other for one second without a kick. A new match clears this fallback. The controller still rejects kicks outside the placement window. Tactical matches use 5% scripted and 95% live self-play before the first snapshot; after snapshots exist, the mix is 5% scripted, 60% live, and 35% frozen tactical. Distillation uses 50% live student self-play and 50% frozen tactical teacher opponents. Joint training uses 50% live joint self-play, 12.5% frozen tactical teachers, and 37.5% frozen joint snapshots once both pools exist; if a frozen pool is empty, its share goes to available opponents. Every 200 updates, a frozen policy enters the pool; the latest eight of each type are retained. Opponents are selected per match. Frozen or scripted matches alone contribute to the recent match-point score used for `model_best.pt`, because points from the live policy playing itself do not measure improvement. Compare candidates with the fixed-seed evaluator before final selection.

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

Evaluation reports learner and opponent goal-door forfeits as well as goals and match points. Scoreless matches count as draws after 30 seconds, so evaluation completes even when neither policy scores or forfeits. Video inspection is available from the evaluator's optional recording flag. A genuine benchmark requires trained skill and football checkpoints; the synthetic-checkpoint smoke test only validates the pipeline.

## Play against a learned policy

Launch the native MuJoCo viewer while controlling one Microduck through the same tactical walking and right-foot kick adapter used during training:

```bash
uv run football-play \
  --opponent-checkpoint logs/football/joint/seed-1/model_best.pt \
  --side home
```

A tactical, distilled, or joint football checkpoint can be the opponent. Tactical opponents use their frozen walking, right-foot kick, and recovery skills; distilled and joint opponents act directly in 14-joint space. Its run directory must contain the copied `skills/` bundle. The human controls one robot; the learned checkpoint controls the other. The selected opponent and `--side home|away|random` setting persist across matches. Attack direction still changes between matches and is printed in the terminal.

Keep the launching **terminal focused** for controls while watching the viewer window. Arrow Up/Down increase/decrease forward velocity; Left/Right strafe; A/E turn left/right. Commands persist until changed. Space stops movement and cancels a queued kick. L queues one right-foot kick for up to two seconds; it starts when the ball enters the kick zone and then runs its 0.6-second skill sequence. Q quits. The viewer can be closed to exit. The terminal must support single-key input (TTY).

For a noninteractive pipeline check, use `--headless --steps 24`. This sends a zero tactical command and does not open a viewer.
