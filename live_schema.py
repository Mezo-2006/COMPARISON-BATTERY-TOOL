"""Payload schema and parser for live (MQTT) mode.

The live path streams samples over MQTT into the same comparison
pipeline used for offline CSV files. This module owns the *contract*
between a publisher (the Digital Twin simulator and the ECU firmware)
and the rest of the tool: it knows how to

1. route a topic to a source label (``"twin"`` or ``"ecu"``) by matching
   the message topic against the two topic *roots* the user configured,
2. parse the JSON payload into a canonical dict of
   ``{timestamp, voltage, current, temperature, soc, soh}`` using the
   same alias-aware column resolution the CSV loader uses, and
3. validate the result, raising ``LiveSchemaError`` on any contract
   violation so the caller can drop the bad packet instead of poisoning
   the accumulator.

The module is **pure Python** (no Qt, no paho) so it can be unit-tested
straight from the command line. It is the live-mode counterpart of
``data_loader._resolve_columns`` and ``data_loader._normalise_timestamp``.


Payload shapes
--------------
Two JSON shapes are supported. The publisher picks one per message.

**Single sample** — one point in time, partial signals allowed::

    {"t": 12.34, "v": 3.65, "i": -1.2, "temp": 28.1, "soc": 78.3, "soh": 99.0}

**Batch** — multiple points, e.g. a 1-second window of 10 Hz samples::

    {"samples": [
        {"t": 12.30, "v": 3.65, "i": -1.2, "temp": 28.0, "soc": 78.2, "soh": 99.0},
        {"t": 12.40, "v": 3.64, "i": -1.1, "temp": 28.1, "soc": 78.3, "soh": 99.0},
    ]}

Keys are matched case-insensitively against the alias groups below; the
same groups the CSV loader uses, so a field called ``voltage`` is just
as valid as ``v`` or ``vbat``. An unknown key is ignored (a warning could
be raised, but the live path favours throughput over diagnostics — the
accumulator surfaces what got through).

Timestamp (``t`` / ``time`` / ``timestamp`` / ...) is mandatory per
sample. Every other signal is optional; the accumulator simply records
``NaN`` for any signal a sample doesn't carry.


Deduplication id (QoS 1)
------------------------
MQTT QoS 1 guarantees **at-least-once** delivery: the broker may
redeliver a message if an acknowledgement is lost. To keep the
accumulator from double-counting a redelivered sample, the publisher
stamps each logical sample with a unique ``id``. Accepted id keys
(case-insensitive): ``id``, ``seq``, ``seq_id``, ``msg_id``,
``message_id``. The id is stringified and carried on the
:class:`LiveSample`; the accumulator drops any sample whose id is
already in the buffer. When the payload carries no id key (``id is
None``), the accumulator falls back to timestamp-based dedup.


Topic routing
-------------
The user configures two topic *roots* in the Live tab — one for the
twin stream and one for the ECU stream. A message's topic is matched
against these roots with MQTT-semantics:

  - ``bms/twin/#`` matches ``bms/twin``, ``bms/twin/data``,
    ``bms/twin/a/b``, and so on (``#`` is the multi-level wildcard).
  - ``bms/twin`` (no wildcard) matches only the exact topic
    ``bms/twin``; a trailing slash is part of the topic, not a
    wildcard, so ``bms/twin/`` matches only ``bms/twin/``.

The match is intentionally simple — substring-style on the root with the
``#`` stripped — because the subscriber in ``MqttWorker`` already filters
by these topics at the broker (via paho ``subscribe``). This check is
just a safety net for the rare case where the user subscribes to a
shared parent topic that catches both streams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_loader import _ALIAS_MAP


# ---------------------------------------------------------------------------
# Canonical signal column ordering — kept in sync with data_loader.
# ---------------------------------------------------------------------------
_CANONICAL_COLUMNS: list[str] = [
    "timestamp", "voltage", "current", "temperature", "soc", "soh",
]

# Accepted JSON keys (case-insensitive, stripped) for the per-sample
# deduplication id.  The publisher stamps each logical sample with a
# unique id so the accumulator can drop QoS-1 redeliveries; without one
# the accumulator falls back to timestamp-based dedup.
_ID_ALIASES: list[str] = [
    "id", "seq", "seq_id", "msg_id", "message_id",
]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------
class LiveSchemaError(Exception):
    """Fatal error while parsing a live MQTT payload.

    Raised for: malformed JSON / non-object payload, a sample without a
    timestamp field, a timestamp value that cannot be coerced to float,
    or a batch payload whose ``samples`` list is missing / not a list.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class LiveSample:
    """One parsed MQTT message, ready to hand to the accumulator.

    Attributes
    ----------
    source_label
        ``"twin"`` or ``"ecu"`` — which stream this sample belongs to.
    columns
        Mapping of canonical column name → value, e.g.
        ``{"timestamp": 12.34, "voltage": 3.65, "current": -1.2,
        "temperature": 28.1, "soc": 78.3, "soh": 99.0}``.  Any canonical
        signal not carried by the sample is simply absent from the dict
        (the accumulator fills ``NaN``).
    source_keys
        The original JSON keys as they arrived on the wire (diagnostics).
    id
        Optional publisher-assigned deduplication id (QoS-1 at-least-once
        delivery can redeliver a message; the accumulator drops a sample
        whose ``id`` is already in the buffer).  ``None`` when the payload
        carries no id key, in which case the accumulator falls back to
        timestamp-based dedup.  The id is stored as a string so int and
        str ids from different publishers don't collide on dtype.
    """

    source_label: str
    columns: Dict[str, float]
    source_keys: List[str] = field(default_factory=list)
    id: Optional[str] = None

    @property
    def timestamp(self) -> float:
        """The sample's timestamp (canonical key ``"timestamp"``)."""
        return self.columns["timestamp"]


# ---------------------------------------------------------------------------
# Topic → source-label routing
# ---------------------------------------------------------------------------
def resolve_source(topic: str, twin_root: str, ecu_root: str) -> str:
    """Map an MQTT topic to a source label by matching against two roots.

    Parameters
    ----------
    topic
        The message topic (e.g. ``"bms/twin/data"``).
    twin_root, ecu_root
        The two topic roots the user configured.  A trailing ``/#`` or
        ``#`` is treated as a multi-level wildcard prefix.

    Returns
    -------
    str
        ``"twin"`` if ``topic`` matches ``twin_root``, ``"ecu"`` if it
        matches ``ecu_root``.

    Raises
    ------
    LiveSchemaError
        If the topic matches neither root (shouldn't happen in normal
        operation — paho has already filtered by the subscription — but
        it guards against a mis-configured shared subscription).
    """
    if _topic_matches(topic, twin_root):
        return "twin"
    if _topic_matches(topic, ecu_root):
        return "ecu"
    raise LiveSchemaError(
        f"Topic {topic!r} matches neither twin root {twin_root!r} nor "
        f"ECU root {ecu_root!r}."
    )


def _topic_matches(topic: str, root: str) -> bool:
    """Return True if ``topic`` falls under the MQTT-filter ``root``.

    Handles the two wildcard cases the UI offers:
      - ``root`` ending in ``/#`` → prefix match on the part before ``/#``.
      - ``root`` ending in ``/`` → exact prefix match.
      - ``root`` with no wildcard → exact equality (``#``-only roots match
        anything but are unusual).

    Examples
    --------
    >>> _topic_matches("bms/twin/data", "bms/twin/#")
    True
    >>> _topic_matches("bms/twin", "bms/twin/#")
    True
    >>> _topic_matches("bms/actual/data", "bms/twin/#")
    False
    >>> _topic_matches("bms/twin/data", "bms/twin/")
    False
    >>> _topic_matches("bms/twin/", "bms/twin/")
    True
    """
    root = root.strip()
    if root.endswith("/#"):
        prefix = root[:-2]                # strip "/#"
        # "bms/twin/#" matches both "bms/twin" and "bms/twin/...".
        return topic == prefix or topic.startswith(prefix + "/")
    if root.endswith("#"):
        prefix = root[:-1]
        return topic.startswith(prefix)
    # No wildcard — exact topic match (trailing slash is part of the
    # topic, not a wildcard).  "bms/twin/" matches only "bms/twin/".
    return topic == root


# ---------------------------------------------------------------------------
# One-sample column resolution
# ---------------------------------------------------------------------------
def _resolve_sample_columns(
    sample: Dict[str, Any],
) -> Dict[str, float]:
    """Normalise one JSON-sample dict into canonical column → value.

    Mirrors ``data_loader._resolve_columns`` but for a single dict: each
    key (lower-cased, stripped) is matched against the alias groups in
    :data:`data_loader._ALIAS_MAP` and the first match per canonical
    name wins.  Unknown keys are ignored.  The special ``t`` ambiguity
    rule is mirrored too: ``t`` resolves to timestamp if a ``v``/i.e.
    voltage alias is also present, otherwise to temperature.

    Timestamp coercion to float is mandatory (raises
    :class:`LiveSchemaError` on failure); any other signal that fails to
    coerce becomes ``NaN`` so the accumulator can still receive the
    timestamp + other valid signals of the same sample.
    """
    lower_to_original: Dict[str, str] = {}
    for k in sample:
        key = k.strip().lower()
        if key not in lower_to_original:
            lower_to_original[key] = k

    canonical_map: Dict[str, str] = {}  # canonical -> original key
    used: set[str] = set()

    for canon, aliases in _ALIAS_MAP.items():
        for alias in aliases:
            if canon == "timestamp" and alias == "id":
                continue
            if alias in lower_to_original and canon not in used:
                canonical_map[canon] = lower_to_original[alias]
                used.add(canon)
                break

    # Single-letter ``t`` ambiguity (mirror of data_loader rule, but
    # generalised to "any voltage alias matched" rather than only ``v``):
    #   - if a voltage signal was found via any alias, ``t`` is the
    #     timestamp (the common twin layout);
    #   - otherwise ``t`` is temperature.
    if "t" in lower_to_original and "timestamp" not in used \
       and "temperature" not in used:
        if "voltage" in used:
            canonical_map["timestamp"] = lower_to_original["t"]
            used.add("timestamp")
        else:
            canonical_map["temperature"] = lower_to_original["t"]
            used.add("temperature")

    if "timestamp" not in used and "id" in lower_to_original:
        canonical_map["timestamp"] = lower_to_original["id"]
        used.add("timestamp")

    out: Dict[str, float] = {}

    # Timestamp — mandatory, hard-fail if absent or uncoercible.
    if "timestamp" not in canonical_map:
        raise LiveSchemaError(
            "Sample has no timestamp field. Accepted keys: "
            f"{_ALIAS_MAP['timestamp']}"
        )
        return out  # unreachable; satisfies type-checkers

    t_raw = sample[canonical_map["timestamp"]]
    try:
        t = float(t_raw)
    except (TypeError, ValueError):
        raise LiveSchemaError(
            f"Timestamp value {t_raw!r} could not be coerced to float."
        )
    if t != t:  # NaN check
        raise LiveSchemaError("Timestamp resolved to NaN.")
    out["timestamp"] = t

    # Signals — best-effort, NaN on coercion failure.
    for canon in ("voltage", "current", "temperature", "soc", "soh"):
        if canon not in canonical_map:
            continue
        raw = sample[canonical_map[canon]]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = float("nan")
        out[canon] = v
    return out


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------
def parse_message(
    topic: str,
    payload: bytes,
    twin_root: str,
    ecu_root: str,
) -> List[LiveSample]:
    """Parse one MQTT message into a list of ``LiveSample`` objects.

    Returns a list because a batch payload carries many samples. The
    caller (the live accumulator) iterates the list and feeds each
    ``LiveSample`` into the appropriate per-source buffer.

    Parameters
    ----------
    topic
        The MQTT topic the message arrived on.
    payload
        Raw MQTT payload bytes (JSON-encoded).
    twin_root, ecu_root
        Topic roots used to route the message to a source label.

    Returns
    -------
    list[LiveSample]
        One :class:`LiveSample` per timestamp in the payload (a batch
        yields many). Empty list only if the payload is an empty batch
        (``{"samples": []}``) — callers should still tolerate that.

    Raises
    ------
    LiveSchemaError
        If the JSON is malformed, the top-level object is not a dict,
        the timestamp is missing/uncoercible, or a ``samples`` field is
        present but not a list.
    """
    source_label = resolve_source(topic, twin_root, ecu_root)

    # --- Decode JSON ------------------------------------------------------
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LiveSchemaError(f"Payload is not valid UTF-8: {exc}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LiveSchemaError(f"Payload is not valid JSON: {exc}")

    if not isinstance(obj, dict):
        raise LiveSchemaError(
            f"Payload top level must be a JSON object, got {type(obj).__name__}."
        )

    # --- Branch: batch vs. single -----------------------------------------
    if "samples" in obj:
        samples_raw = obj["samples"]
        if not isinstance(samples_raw, list):
            raise LiveSchemaError(
                "'samples' field must be a list of sample objects."
            )
        # Empty batch is valid — return no samples (caller skips).
        out: List[LiveSample] = []
        for sample in samples_raw:
            if not isinstance(sample, dict):
                raise LiveSchemaError(
                    "Each entry in 'samples' must be a JSON object."
                )
            out.append(_build_sample(source_label, sample))
        return out

    # Single sample.
    return [_build_sample(source_label, obj)]


def _build_sample(source_label: str, sample: Dict[str, Any]) -> LiveSample:
    """Build one ``LiveSample`` from a single-sample dict."""
    columns = _resolve_sample_columns(sample)
    sample_id = _resolve_id(sample)
    return LiveSample(
        source_label=source_label,
        columns=columns,
        source_keys=list(sample.keys()),
        id=sample_id,
    )


def _resolve_id(sample: Dict[str, Any]) -> Optional[str]:
    """Extract the optional deduplication id from a sample dict.

    Matches the first key (case-insensitive, stripped) in
    :data:`_ID_ALIASES`.  The value is stringified so an int ``id`` from
    one publisher and a str ``id`` from another don't collide on pandas
    dtype when mixed in the same buffer.  Returns ``None`` when the
    sample carries no id key (the accumulator then falls back to
    timestamp-based dedup).
    """
    lower_to_original: Dict[str, str] = {}
    for k in sample:
        key = k.strip().lower()
        if key not in lower_to_original:
            lower_to_original[key] = k
    for alias in _ID_ALIASES:
        if alias in lower_to_original:
            val = sample[lower_to_original[alias]]
            if val is None:
                return None
            return str(val)
    return None