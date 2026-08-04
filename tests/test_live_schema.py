"""Unit tests for ``live_schema.py`` — the live MQTT payload parser.

These tests are pure Python: no Qt, no paho. They feed raw bytes into
:func:`live_schema.parse_message` and assert on the resulting
:class:`LiveSample` list. All happy-path, error, and routing cases are
covered.
"""

from __future__ import annotations

import json

import pytest

from live_schema import (
    LiveSample,
    LiveSchemaError,
    parse_message,
    resolve_source,
    _topic_matches,
)


TWIN_ROOT = "bms/twin/#"
ECU_ROOT = "bms/actual/#"


# ---------------------------------------------------------------------------
# Payload encoding helper
# ---------------------------------------------------------------------------
def _payload(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


# ---------------------------------------------------------------------------
# resolve_source — topic routing
# ---------------------------------------------------------------------------
class TestResolveSource:
    def test_twin_topic_matches_twin_root(self):
        assert resolve_source("bms/twin/data", TWIN_ROOT, ECU_ROOT) == "twin"

    def test_ecu_topic_matches_ecu_root(self):
        assert resolve_source("bms/actual/data", TWIN_ROOT, ECU_ROOT) == "ecu"

    def test_twin_root_exact_match_without_slash(self):
        # "bms/twin/#" also matches the exact "bms/twin" (no slash).
        assert resolve_source("bms/twin", TWIN_ROOT, ECU_ROOT) == "twin"

    def test_nested_topic_matches(self):
        assert resolve_source("bms/twin/a/b/c", TWIN_ROOT, ECU_ROOT) == "twin"

    def test_no_match_raises(self):
        with pytest.raises(LiveSchemaError):
            resolve_source("other/topic", TWIN_ROOT, ECU_ROOT)


# ---------------------------------------------------------------------------
# _topic_matches — wildcard semantics
# ---------------------------------------------------------------------------
class TestTopicMatches:
    def test_multilevel_wildcard_prefix_match(self):
        assert _topic_matches("bms/twin/data", "bms/twin/#") is True

    def test_multilevel_wildcard_exact_match_no_slash(self):
        # "bms/twin/#" matches "bms/twin" (the part before "/#").
        assert _topic_matches("bms/twin", "bms/twin/#") is True

    def test_multilevel_wildcard_non_match(self):
        assert _topic_matches("bms/actual/data", "bms/twin/#") is False

    def test_trailing_slash_exact_match(self):
        # A root with no wildcard matches only the exact topic — the
        # trailing slash is part of the topic, not a prefix wildcard.
        assert _topic_matches("bms/twin/", "bms/twin/") is True
        assert _topic_matches("bms/twin/data", "bms/twin/") is False

    def test_no_wildcard_exact_equality(self):
        assert _topic_matches("bms/twin", "bms/twin") is True
        assert _topic_matches("bms/twin/data", "bms/twin") is False


# ---------------------------------------------------------------------------
# parse_message — single-sample happy paths
# ---------------------------------------------------------------------------
class TestParseSingleSample:
    def test_minimal_short_keys(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 12.34, "v": 3.65}), TWIN_ROOT, ECU_ROOT,
        )
        assert len(samples) == 1
        s = samples[0]
        assert s.source_label == "twin"
        assert s.columns["timestamp"] == 12.34
        assert s.columns["voltage"] == 3.65
        # Signals not present in the payload are absent from columns.
        assert "current" not in s.columns
        assert "temperature" not in s.columns

    def test_full_long_keys(self):
        samples = parse_message(
            "bms/actual",
            _payload({
                "timestamp": 1.0, "voltage": 3.7, "current": -1.5,
                "temperature": 30.0, "soc": 80.0, "soh": 99.0,
            }),
            TWIN_ROOT, ECU_ROOT,
        )
        assert len(samples) == 1
        s = samples[0]
        assert s.source_label == "ecu"
        # Every canonical signal is present.
        assert set(s.columns) == {
            "timestamp", "voltage", "current", "temperature", "soc", "soh",
        }
        assert s.columns["soc"] == 80.0

    def test_alias_v_vbat(self):
        # ``v`` and ``vbat`` both map to voltage; first match wins but
        # they're in the same group so only one shows up.
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "vbat": 3.3}), TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].columns["voltage"] == 3.3

    def test_timestamp_mandatory(self):
        with pytest.raises(LiveSchemaError):
            parse_message(
                "bms/twin", _payload({"v": 3.3}), TWIN_ROOT, ECU_ROOT,
            )


# ---------------------------------------------------------------------------
# parse_message — batch happy paths
# ---------------------------------------------------------------------------
class TestParseBatch:
    def test_batch_yields_multiple_samples(self):
        samples = parse_message(
            "bms/twin",
            _payload({"samples": [
                {"t": 0.0, "v": 3.3},
                {"t": 0.1, "v": 3.4},
                {"t": 0.2, "v": 3.5},
            ]}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert len(samples) == 3
        assert [s.timestamp for s in samples] == [0.0, 0.1, 0.2]
        assert all(s.source_label == "twin" for s in samples)

    def test_empty_batch_returns_empty_list(self):
        samples = parse_message(
            "bms/twin", _payload({"samples": []}), TWIN_ROOT, ECU_ROOT,
        )
        assert samples == []

    def test_batch_samples_must_be_list(self):
        with pytest.raises(LiveSchemaError):
            parse_message(
                "bms/twin", _payload({"samples": "not a list"}),
                TWIN_ROOT, ECU_ROOT,
            )

    def test_batch_entry_must_be_object(self):
        with pytest.raises(LiveSchemaError):
            parse_message(
                "bms/twin", _payload({"samples": [1, 2, 3]}),
                TWIN_ROOT, ECU_ROOT,
            )


# ---------------------------------------------------------------------------
# parse_message — error cases
# ---------------------------------------------------------------------------
class TestParseErrors:
    def test_malformed_json_raises(self):
        with pytest.raises(LiveSchemaError):
            parse_message("bms/twin", b"not json{", TWIN_ROOT, ECU_ROOT)

    def test_non_utf8_payload_raises(self):
        with pytest.raises(LiveSchemaError):
            parse_message("bms/twin", b"\xff\xfe", TWIN_ROOT, ECU_ROOT)

    def test_top_level_array_raises(self):
        with pytest.raises(LiveSchemaError):
            parse_message("bms/twin", _payload([1, 2, 3]), TWIN_ROOT, ECU_ROOT)

    def test_non_numeric_timestamp_raises(self):
        with pytest.raises(LiveSchemaError):
            parse_message(
                "bms/twin", _payload({"t": "abc"}), TWIN_ROOT, ECU_ROOT,
            )

    def test_non_numeric_signal_becomes_nan(self):
        # A non-numeric signal value should NOT crash — it becomes NaN
        # so the accumulator still records the timestamp + other valid
        # signals of the same sample.
        samples = parse_message(
            "bms/twin",
            _payload({"t": 1.0, "v": "bad"}),
            TWIN_ROOT, ECU_ROOT,
        )
        import math
        assert math.isnan(samples[0].columns["voltage"])

    def test_nan_timestamp_raises(self):
        # float('nan') is a float but should be rejected (would break
        # the alignment engine's sort + overlap checks).
        with pytest.raises(LiveSchemaError):
            parse_message(
                "bms/twin", _payload({"t": "nan"}), TWIN_ROOT, ECU_ROOT,
            )


# ---------------------------------------------------------------------------
# LiveSample dataclass
# ---------------------------------------------------------------------------
class TestLiveSampleDataclass:
    def test_timestamp_property(self):
        s = LiveSample(
            source_label="twin",
            columns={"timestamp": 5.5, "voltage": 3.3},
        )
        assert s.timestamp == 5.5

    def test_source_keys_default_empty(self):
        s = LiveSample("twin", {"timestamp": 0.0})
        assert s.source_keys == []

    def test_id_defaults_none(self):
        s = LiveSample("twin", {"timestamp": 0.0})
        assert s.id is None


# ---------------------------------------------------------------------------
# QoS-1 deduplication id
# ---------------------------------------------------------------------------
class TestResolveId:
    def test_id_key_extracted(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "v": 3.3, "id": "abc-1"}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].id == "abc-1"

    def test_seq_alias_extracted(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "v": 3.3, "seq": 42}),
            TWIN_ROOT, ECU_ROOT,
        )
        # int id is stringified so mixed int/str ids don't collide on dtype.
        assert samples[0].id == "42"

    def test_msg_id_alias_extracted(self):
        samples = parse_message(
            "bms/actual",
            _payload({"t": 0.0, "v": 3.3, "msg_id": "m-99"}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].id == "m-99"

    def test_id_case_insensitive(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "v": 3.3, "ID": "x"}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].id == "x"

    def test_no_id_yields_none(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "v": 3.3}), TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].id is None

    def test_id_null_value_yields_none(self):
        samples = parse_message(
            "bms/twin", _payload({"t": 0.0, "v": 3.3, "id": None}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert samples[0].id is None

    def test_batch_samples_each_carry_own_id(self):
        samples = parse_message(
            "bms/twin",
            _payload({"samples": [
                {"t": 0.0, "v": 3.3, "id": "a"},
                {"t": 0.1, "v": 3.4, "id": "b"},
            ]}),
            TWIN_ROOT, ECU_ROOT,
        )
        assert [s.id for s in samples] == ["a", "b"]