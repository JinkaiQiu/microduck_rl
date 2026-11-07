"""Failure-only reset reports, retaining raw state across the reset boundary."""

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import mujoco
import torch


class ResetDiagnostics:
    def __init__(self, arena, directory):
        self.arena = arena
        self.directory = Path(directory)
        self.context = {}
        self.context_provider = None
        self.after_events = None

    def capture(self, ids):
        """Advanced indexing copies on device; healthy resets do no file I/O."""
        arena = self.arena
        values = {}
        for prefix, source, names in (
            ("data", arena.env.sim.data, (
                "qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "act_dot",
                "qfrc_applied", "xfrc_applied", "qfrc_bias", "qfrc_constraint",
                "qfrc_actuator", "qfrc_passive", "sensordata", "site_xpos",
                "cvel", "cdof_dot", "cacc", "time", "nefc", "solver_niter", "eq_active",
            )),
            ("model", getattr(arena.env.sim, "model", None), ("dof_frictionloss", "dof_damping")),
        ):
            for name in names:
                value = getattr(source, name, None)
                if value is not None and value.ndim and value.shape[0] in (1, arena.num_envs):
                    values[f"{prefix}/{name}"] = value.expand((arena.num_envs,) + value.shape[1:])[ids]
        for name, value in (
            ("elapsed", arena.elapsed), ("dones", arena.dones),
            ("previous_vel", arena._previous_vel), ("last_actions", arena._last_actions),
            ("actor_cache", arena._actor_cache), ("critic_cache", arena._critic_cache),
        ):
            values[name] = value[ids]
        for side in ("robot", "opponent"):
            for index, actuator in enumerate(getattr(arena.env.scene[side], "actuators", ())):
                value = getattr(actuator, "_prev_motor_torque", None)
                if value is not None:
                    values[f"actuator/{side}/{index}/previous_motor_torque"] = value[ids]
        return values

    @staticmethod
    def _summary(value):
        flat = value.flatten().double()
        finite = torch.isfinite(flat)
        valid = flat[finite]
        return {
            "shape": list(value.shape), "nonfinite_count": int((~finite).sum()),
            "above_mjmaxval_count": int((flat.abs() > mujoco.mjMAXVAL).sum()),
            "finite_max_abs": float(valid.abs().max()) if valid.numel() else None,
            "nonfinite_indices": (~torch.isfinite(value)).nonzero()[:16].tolist(),
            "above_mjmaxval_indices": (flat.reshape(value.shape).abs() > mujoco.mjMAXVAL).nonzero()[:16].tolist(),
        }

    def write(self, stage, ids, before, checks=None, error=None):
        """Bound dump size; a report failure must never mask the original error."""
        try:
            return self._write(stage, ids, before, checks or {}, error)
        except Exception as exc:
            print(f"[tug diagnostics] report failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _write(self, stage, ids, before, checks, error):
        arena = self.arena
        all_ids = ids.cpu().tolist()
        failed = torch.zeros(arena.num_envs, dtype=torch.bool, device=arena.device)
        for mask in checks.values():
            failed |= mask
        failed_ids = failed.nonzero().flatten().cpu().tolist()
        # Include every requested/failed ID in JSON, but at most eight raw worlds.
        selected = [index for index in failed_ids if index in all_ids] or all_ids
        selected = selected[:8]
        selected_ids = torch.tensor(selected, device=arena.device, dtype=torch.long)
        rows = torch.tensor([all_ids.index(index) for index in selected], device=arena.device, dtype=torch.long)
        snapshots = {"before_reset": {name: value[rows].cpu() for name, value in before.items()},
                     "after_forward": {name: value.cpu() for name, value in self.capture(selected_ids).items()}}
        if self.after_events is not None:
            snapshots["after_reset_events"] = {name: value[rows].cpu() for name, value in self.after_events.items()}
        from microduck_challenges.tug_of_war import env as reference

        report = {
            "schema": 1, "stage": stage, "error": error,
            "utc": datetime.now(timezone.utc).isoformat(), "common_step_counter": int(getattr(arena.env, "common_step_counter", 0)),
            "requested_env_ids": all_ids, "failed_env_ids": failed_ids, "dumped_env_ids": selected,
            "checks": {name: mask.nonzero().flatten().cpu().tolist() for name, mask in checks.items() if mask.any()},
            "thresholds": {"mjMAXVAL": mujoco.mjMAXVAL, "tie_distance_m": .05,
                           "string_length_m": reference.STRING_M},
            "context": {**self.context, **(self.context_provider(selected) if self.context_provider else {})},
            "expanded_model_fields": sorted(getattr(arena.env.sim, "expanded_fields", ())),
            "snapshots": {},
        }
        for name, values in snapshots.items():
            sites = values["data/site_xpos"]
            ends = sites[:, [arena._rope_ids[0], arena._rope_ids[-1]]]
            necks = sites[:, arena._neck_ids]
            distances = (ends - necks).norm(dim=-1)
            report["snapshots"][name] = {
                "derived_fields_stale": name == "after_reset_events",
                "fields": {key: self._summary(value) for key, value in values.items()},
                "tie_distances_m": [[float(x) if torch.isfinite(x) else None for x in row] for row in distances],
                "neck_separation_m": [float(x) if torch.isfinite(x) else None for x in (necks[:, 1] - necks[:, 0]).norm(dim=-1)],
            }
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        path = self.directory / stem
        model_path = self.directory / "model.mjb"
        if not model_path.exists() and isinstance(arena.env.sim.mj_model, mujoco.MjModel):
            mujoco.mj_saveModel(arena.env.sim.mj_model, str(model_path))
        torch.save({"env_ids": selected, "snapshots": snapshots,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path.with_suffix(".pt"))
        path.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[tug diagnostics] {stage}: checks={report['checks']} report={path.with_suffix('.json')}", flush=True)
        return path.with_suffix(".json")
