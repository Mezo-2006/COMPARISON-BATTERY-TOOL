"""Seeded synthetic twin/ECU battery telemetry generator for live-mode testing.

This module produces JSON payloads shaped exactly like ``live_schema.py``
expects on the wire — the "batch" shape (``{"samples": [...]}``) with
canonical-alias keys and a per-sample ``id`` that is a small, increasing
counter (starting from 0 by default). The payload intentionally does not
include a separate ``timestamp`` key.

Three independent, seed-derived sources of randomness
-------------------------------------------------------
A single :class:`GeneratorConfig.seed` is split (via
``numpy.random.SeedSequence.spawn``) into three child RNGs, one per axis
of randomness, so a given seed always reproduces byte-identical twin+ecu
JSON and each axis can be reasoned about / tuned independently:

1. ``rng_sensor``      — per-sample measurement noise, applied
   independently to the twin and the ecu reading of the same true value
   (simulates ADC/sensor jitter). Kept small so both streams stay in a
   physically normal operating range.
2. ``rng_divergence``  — the twin-vs-ecu model gap: a small bias +
   noise added on top of the ecu reading so ``|ecu - twin|`` stays close
   to :data:`GeneratorConfig.divergence_bias` (~0.1 in each signal's own
   unit by default) — comfortably inside the tool's default tolerance
   floors (see ``statistics_engine._ABS_FLOOR``), so a default run should
   read as a healthy "match".
3. ``rng_trajectory``  — a slow random walk applied to the underlying
   drift rates (discharge ramp, temperature rise, ...) so different seeds
   produce different but still physically plausible discharge curves
   instead of noise around one fixed straight line.

Signal ranges and twin/ecu value divergence match the project's own test
fixtures (``test_twin.csv`` / ``test_ecu.csv`` and
``tests/conftest.py``'s ``twin_result_five`` / ``ecu_result_five``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Canonical signal ordering — kept in sync with data_loader / live_schema.
_SIGNALS: List[str] = ["voltage", "current", "temperature", "soc", "soh"]


@dataclass
class GeneratorConfig:
    """Tunable knobs for one synthetic run. Same seed -> identical output."""

    seed: int = 42
    dt: float = 0.1                       # seconds between twin samples
    ecu_time_offset: float = 0.05         # kept for publisher compatibility
    start_time: float | None = None       # deprecated alias for start_id
    start_id: int = 0                     # first sample id for twin/ecu

    # Starting state (matches the project's CSV fixtures).
    start_voltage: float = 3.30           # V
    start_current: float = 1.20           # A
    start_temperature: float = 25.0       # degC
    start_soc: float = 80.0               # %
    start_soh: float = 95.0               # %

    # Nominal drift of the *true* (twin) trajectory, per second.
    voltage_drift_per_s: float = 0.10     # V/s, discharge ramp (falls)
    current_drift_per_s: float = 0.0      # A/s, roughly steady load
    temperature_drift_per_s: float = 1.0  # degC/s, gentle warm-up
    soc_drain_per_s: float = 1.0          # %/s
    soh_drift_per_s: float = 0.0          # %/s, ~flat over a short run

    # 1) Sensor noise std-dev, per signal unit.
    sensor_noise_std: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.004, "current": 0.01, "temperature": 0.03,
        "soc": 0.02, "soh": 0.005,
    })

    # 2) Twin/ECU divergence bias + spread (target |ecu - twin| ~ 0.1).
    divergence_bias: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.005, "current": 0.005, "temperature": 0.05,
        "soc": 0.05, "soh": 0.0,
    })
    divergence_std: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.01, "current": 0.01, "temperature": 0.05,
        "soc": 0.05, "soh": 0.02,
    })

    # 3) Slow random walk applied to the drift rates themselves.
    trajectory_walk_std: Dict[str, float] = field(default_factory=lambda: {
        "voltage": 0.0005, "current": 0.001, "temperature": 0.005,
        "soc": 0.01, "soh": 0.0005,
    })

    # Physical clamps — never emit outside these regardless of accumulated
    # noise/walk.
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "voltage": (2.8, 4.2), "current": (0.0, 5.0),
        "temperature": (0.0, 60.0), "soc": (0.0, 100.0), "soh": (0.0, 100.0),
    })


class SyntheticBatteryGenerator:
    """Generates seeded, ``live_schema``-compliant twin/ECU sample pairs."""

    def __init__(self, config: GeneratorConfig | None = None):
        self.cfg = config or GeneratorConfig()

        seed_seq = np.random.SeedSequence(self.cfg.seed)
        s_sensor, s_divergence, s_trajectory = seed_seq.spawn(3)
        self.rng_sensor = np.random.default_rng(s_sensor)
        self.rng_divergence = np.random.default_rng(s_divergence)
        self.rng_trajectory = np.random.default_rng(s_trajectory)

        if self.cfg.start_time is not None:
            self._seq = int(self.cfg.start_time)
        else:
            self._seq = int(self.cfg.start_id)
        self._true = {
            "voltage": self.cfg.start_voltage,
            "current": self.cfg.start_current,
            "temperature": self.cfg.start_temperature,
            "soc": self.cfg.start_soc,
            "soh": self.cfg.start_soh,
        }
        self._drift_per_s = {
            "voltage": -self.cfg.voltage_drift_per_s,   # discharging -> falls
            "current": self.cfg.current_drift_per_s,
            "temperature": self.cfg.temperature_drift_per_s,
            "soc": -self.cfg.soc_drain_per_s,
            "soh": -self.cfg.soh_drift_per_s,
        }

    def _clamp(self, name: str, value: float) -> float:
        lo, hi = self.cfg.bounds[name]
        return float(min(max(value, lo), hi))

    def step(self) -> Tuple[dict, dict]:
        """Advance one ``dt`` and return ``(twin_sample, ecu_sample)`` dicts."""
        cfg = self.cfg
        dt = cfg.dt

        # 3) trajectory randomness: nudge each drift rate with a slow walk.
        for name in _SIGNALS:
            self._drift_per_s[name] += self.rng_trajectory.normal(
                0.0, cfg.trajectory_walk_std[name])

        true_now = {}
        for name in _SIGNALS:
            self._true[name] = self._clamp(
                name, self._true[name] + self._drift_per_s[name] * dt)
            true_now[name] = self._true[name]

        # 1) sensor noise — independent draw for twin and for ecu.
        twin = {
            name: self._clamp(name, true_now[name] + self.rng_sensor.normal(
                0.0, cfg.sensor_noise_std[name]))
            for name in _SIGNALS
        }

        # 2) twin/ecu divergence — bias + noise added on top of ecu's own
        #    sensor draw, so |ecu - twin| stays close to divergence_bias.
        ecu = {}
        for name in _SIGNALS:
            ecu_sensor_draw = self.rng_sensor.normal(0.0, cfg.sensor_noise_std[name])
            gap = self.rng_divergence.normal(
                cfg.divergence_bias[name], cfg.divergence_std[name])
            ecu[name] = self._clamp(name, true_now[name] + ecu_sensor_draw + gap)

        sample_id = self._seq
        twin["id"] = sample_id
        ecu["id"] = sample_id
        self._seq += 1
        return twin, ecu

    def batch(self, n: int) -> Tuple[dict, dict]:
        """Return ``n`` steps as two ``live_schema`` batch payloads.

        Returns
        -------
        (twin_payload, ecu_payload)
            Each shaped ``{"samples": [...]}``, ready for
            ``json.dumps(...).encode("utf-8")`` and publishing on the
            twin / ecu MQTT topic respectively.
        """
        twin_samples, ecu_samples = [], []
        for _ in range(n):
            t, e = self.step()
            twin_samples.append(t)
            ecu_samples.append(e)
        return {"samples": twin_samples}, {"samples": ecu_samples}


def batch_to_json_bytes(payload: dict) -> bytes:
    """Encode a batch payload dict the same way a real publisher would."""
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# CLI — dry-run: print N batches of JSON to stdout without touching MQTT.
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Print seeded synthetic twin/ECU JSON batches to stdout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=3, help="number of batches to print")
    parser.add_argument("--batch-size", type=int, default=5, help="samples per batch")
    args = parser.parse_args()

    gen = SyntheticBatteryGenerator(GeneratorConfig(seed=args.seed))
    for i in range(args.batches):
        twin_payload, ecu_payload = gen.batch(args.batch_size)
        print(f"--- batch {i} : twin ---")
        print(json.dumps(twin_payload, indent=2))
        print(f"--- batch {i} : ecu ---")
        print(json.dumps(ecu_payload, indent=2))


if __name__ == "__main__":
    _main()
