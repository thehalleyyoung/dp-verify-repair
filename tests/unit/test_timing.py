"""Tests for dpcegar.utils.timing – Timer, decorator, stopwatch, registry."""

from __future__ import annotations

import time

import pytest

from dpcegar.utils.timing import (
    Stopwatch,
    TimingRecord,
    TimingRegistry,
    global_registry,
    timed,
    timed_block,
)


# ═══════════════════════════════════════════════════════════════════════════
# TimingRecord
# ═══════════════════════════════════════════════════════════════════════════


class TestTimingRecord:
    """Tests for the TimingRecord data class."""

    def test_default_values(self):
        r = TimingRecord(name="test")
        assert r.calls == 0
        assert r.total_seconds == 0.0
        assert r.min_seconds == float("inf")
        assert r.max_seconds == 0.0

    def test_avg_zero_calls(self):
        r = TimingRecord(name="t")
        assert r.avg_seconds == 0.0

    def test_record_single_observation(self):
        r = TimingRecord(name="t")
        r.record(0.5)
        assert r.calls == 1
        assert r.total_seconds == pytest.approx(0.5)
        assert r.min_seconds == pytest.approx(0.5)
        assert r.max_seconds == pytest.approx(0.5)

    def test_record_multiple_observations(self):
        r = TimingRecord(name="t")
        r.record(0.2)
        r.record(0.8)
        r.record(0.5)
        assert r.calls == 3
        assert r.min_seconds == pytest.approx(0.2)
        assert r.max_seconds == pytest.approx(0.8)
        assert r.avg_seconds == pytest.approx(0.5)

    def test_str_contains_name(self):
        r = TimingRecord(name="my_span")
        r.record(1.0)
        s = str(r)
        assert "my_span" in s
        assert "calls=1" in s


# ═══════════════════════════════════════════════════════════════════════════
# TimingRegistry
# ═══════════════════════════════════════════════════════════════════════════


class TestTimingRegistry:
    """Tests for the timing registry."""

    def test_get_creates_record(self):
        reg = TimingRegistry()
        r = reg.get("new_span")
        assert isinstance(r, TimingRecord)
        assert r.name == "new_span"

    def test_get_returns_same_record(self):
        reg = TimingRegistry()
        a = reg.get("span")
        b = reg.get("span")
        assert a is b

    def test_record_updates_named_record(self):
        reg = TimingRegistry()
        reg.record("x", 0.5)
        reg.record("x", 0.3)
        r = reg.get("x")
        assert r.calls == 2

    def test_clear_removes_all(self):
        reg = TimingRegistry()
        reg.record("a", 0.1)
        reg.record("b", 0.2)
        reg.clear()
        assert len(reg.summary()) == 0

    def test_summary_sorted_by_total(self):
        reg = TimingRegistry()
        reg.record("slow", 1.0)
        reg.record("fast", 0.1)
        summary = reg.summary()
        assert summary[0].name == "slow"

    def test_dump_returns_string(self):
        reg = TimingRegistry()
        reg.record("x", 0.5)
        d = reg.dump()
        assert "Timing Summary" in d
        assert "x" in d


# ═══════════════════════════════════════════════════════════════════════════
# global_registry
# ═══════════════════════════════════════════════════════════════════════════


class TestGlobalRegistry:
    """Tests for the global registry singleton."""

    def test_returns_registry(self):
        reg = global_registry()
        assert isinstance(reg, TimingRegistry)

    def test_returns_same_instance(self):
        a = global_registry()
        b = global_registry()
        assert a is b


# ═══════════════════════════════════════════════════════════════════════════
# timed decorator
# ═══════════════════════════════════════════════════════════════════════════


class TestTimedDecorator:
    """Tests for the @timed decorator."""

    def test_decorated_function_runs(self):
        reg = TimingRegistry()

        @timed("test.add", registry=reg)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_decorated_function_records_timing(self):
        reg = TimingRegistry()

        @timed("test.func", registry=reg)
        def func():
            return 42

        func()
        r = reg.get("test.func")
        assert r.calls == 1
        assert r.total_seconds >= 0

    def test_decorated_function_uses_auto_name(self):
        reg = TimingRegistry()

        @timed(registry=reg)
        def my_function():
            return 1

        my_function()
        summary = reg.summary()
        names = [r.name for r in summary]
        assert any("my_function" in n for n in names)

    def test_decorated_function_preserves_exceptions(self):
        reg = TimingRegistry()

        @timed("test.err", registry=reg)
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()
        # Should still record timing even on exception
        r = reg.get("test.err")
        assert r.calls == 1

    def test_multiple_calls_accumulate(self):
        reg = TimingRegistry()

        @timed("test.multi", registry=reg)
        def noop():
            pass

        for _ in range(5):
            noop()
        r = reg.get("test.multi")
        assert r.calls == 5


# ═══════════════════════════════════════════════════════════════════════════
# timed_block context manager
# ═══════════════════════════════════════════════════════════════════════════


class TestTimedBlock:
    """Tests for the timed_block context manager."""

    def test_records_timing(self):
        reg = TimingRegistry()
        with timed_block("test.block", registry=reg):
            _ = sum(range(100))
        r = reg.get("test.block")
        assert r.calls == 1
        assert r.total_seconds >= 0

    def test_records_on_exception(self):
        reg = TimingRegistry()
        with pytest.raises(RuntimeError):
            with timed_block("test.err_block", registry=reg):
                raise RuntimeError("oops")
        r = reg.get("test.err_block")
        assert r.calls == 1

    def test_nested_blocks(self):
        reg = TimingRegistry()
        with timed_block("outer", registry=reg):
            with timed_block("inner", registry=reg):
                _ = 1 + 1
        assert reg.get("outer").calls == 1
        assert reg.get("inner").calls == 1

    def test_elapsed_is_positive(self):
        reg = TimingRegistry()
        with timed_block("test.pos", registry=reg):
            time.sleep(0.01)
        r = reg.get("test.pos")
        assert r.total_seconds > 0


# ═══════════════════════════════════════════════════════════════════════════
# Stopwatch
# ═══════════════════════════════════════════════════════════════════════════


class TestStopwatch:
    """Tests for the Stopwatch utility."""

    def test_initial_elapsed_is_zero(self):
        sw = Stopwatch()
        assert sw.elapsed == 0.0

    def test_start_returns_self(self):
        sw = Stopwatch()
        result = sw.start()
        assert result is sw

    def test_stop_returns_elapsed(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        elapsed = sw.stop()
        assert elapsed > 0

    def test_stop_without_start_returns_zero(self):
        sw = Stopwatch()
        assert sw.stop() == 0.0

    def test_elapsed_while_running(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        assert sw.elapsed > 0

    def test_elapsed_after_stop(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()
        e = sw.elapsed
        assert e > 0
        # Elapsed should be stable after stop
        time.sleep(0.01)
        assert sw.elapsed == pytest.approx(e, abs=1e-6)

    def test_restart(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()
        first = sw.elapsed
        sw.start()
        time.sleep(0.01)
        sw.stop()
        second = sw.elapsed
        # After restart, elapsed is for the second run
        assert second > 0
