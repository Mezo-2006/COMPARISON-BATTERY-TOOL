"""Seeded synthetic twin/ECU battery telemetry generator for live-mode testing.

This module produces JSON payloads shaped exactly like ``live_schema.py``
expects on the wire — the "batch" shape (``{"samples": [...]}``) with
canonical-alias keys plus a per-sample ``id`` for QoS-1 dedup — for two
correlated streams: the *twin* (ground truth / model) and the *ecu*
(real hardware / "actual"). It exists to exercise the live pipeline
(``live_schema`` -> ``live_accumulator`` -> ``live_controller`` ->
``alignment_engine`` -> ``statistics_engine``) without needing real
Simulink or ECU hardware attached to a broker.

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

Signal ranges and the twin/ecu offset convention (ecu timestamped
``ecu_time_offset`` seconds after twin, ecu ~= twin + a small delta) match
the project's own test fixtures (``test_twin.csv`` / ``test_ecu.csv`` and
``tests/conftest.py``'s ``twin_result_five`` / ``ecu_result_five``).

Timestamp convention — real Unix time, in MILLISECONDS
--------------------------------------------------------
**Wire contract:** every sample's ``timestamp`` field is a real Unix
timestamp expressed in **milliseconds** since the epoch (1970-01-01
00:00:00 UTC) — e.g. ``1754748123456`` for 2025-08-09T12:02:03.456Z
(compare to JavaScript's ``Date.now()`` or Java's
``System.currentTimeMillis()``). It is a ``double``/float on the wire
(JSON has no int64), matching the ``double time`` parameter in the
``variablesToJSON(double time, double voltage, double current, double
soc, double soh)`` signature the ECU/Simulink side builds its payload
with, and the ``double`` returned by ``getTime()`` after parsing one
back with ``JSONtoVariables()``. On the C/embedded side this is
``real_unix_millis = (double)time(NULL) * 1000.0 + sub_second_ms`` (or
an RTC/NTP-synced clock) — **not** ``time(NULL)`` alone (that's seconds)
and **not** a free-running tick counter from process start.

**Why milliseconds:** matches common wire/telemetry convention (JS
``Date.now()``, most JSON telemetry/log formats) and gives integer-ish
timestamps with no ambiguity about how many sub-second digits are
significant. Every part of the live pipeline that compares raw
timestamp deltas is scaled to match this unit:
``live_accumulator.DEFAULT_MAX_AGE_MS`` (buffer aging window, 600 000 =
10 minutes) and ``alignment_engine._MAX_DELTA_T_WARN_MS`` (the
nearest-match gap-warning threshold, 500 = half a second). **A real
ECU/Simulink publisher must also emit milliseconds** — sending seconds
(~1.7e9, 1000x smaller) would make the buffer's 10-minute aging window
trip after 10 000 real seconds (~2.8 hours) instead of 10 minutes, and
would make the alignment gap-warning fire on almost every sample.

**Why real time, not a simulation-relative counter:** earlier versions
of this generator started its internal clock at ``t = 0.0`` and just
incremented it by ``dt`` per sample — a timestamp with no relationship
to the wall clock. That doesn't exercise the tool the way a real
publisher would (a real ECU/Simulink sender stamps each sample with
*when it was actually measured*), so this generator instead seeds its
clock from ``time.time() * 1000`` at construction and still advances it
by ``dt`` (converted to ms) per synthetic sample (see
:meth:`SyntheticBatteryGenerator.step`) — every timestamp it emits is a
genuine, real, present-day Unix time in milliseconds, not an offset from
zero. **A real ECU/Simulink publisher should do the same:** stamp
``time`` with the actual real-world instant the sample was measured (in
ms), not a simulation tick counted from process start.
"""

from __future__ import annotations

import argparse
import json
import time as _time
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
    ecu_time_offset: float = 0.05         # ECU sample offset from twin, seconds
    start_time: float | None = None       # Unix EPOCH MILLISECONDS; None ->
                                           # time.time()*1000 at construction
                                           # (real wall clock). Pin an explicit
                                           # value only when a test needs
                                           # byte-identical output across two
                                           # generator instances. dt /
                                           # ecu_time_offset stay in seconds
                                           # for readability (drift rates are
                                           # "per second") and are converted
                                           # to ms internally -- see step().

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

        # Seed the internal clock from the real wall clock (real Unix
        # epoch MILLISECONDS) rather than 0.0 — see the module docstring's
        # "Timestamp convention" section. Still advances by ``dt`` (in ms,
        # see step()) per sample, so per-sample spacing within/across
        # batches stays deterministic even though the anchor is real.
        # ``start_time`` lets a test pin the anchor for byte-identical-
        # output comparisons.
        self._t = (self.cfg.start_time if self.cfg.start_time is not None
                   else _time.time() * 1000.0)
        self._seq = 0
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

        self._seq += 1
        # Wire timestamp is Unix epoch MILLISECONDS (self._t); dt and
        # ecu_time_offset are authored in seconds for readability
        # (matches the per-second drift rates above), so convert here.
        twin["timestamp"] = round(self._t, 3)
        twin["id"] = self._seq
        ecu["timestamp"] = round(self._t + cfg.ecu_time_offset * 1000.0, 3)
        ecu["id"] = self._seq

        self._t += dt * 1000.0
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
