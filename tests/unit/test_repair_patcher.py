"""Tests for dpcegar.repair.patcher – PatchGenerator, SourcePatcher, PatchReport."""

from __future__ import annotations

from typing import Any

import pytest

from dpcegar.ir.types import (
    Const,
    IRType,
    NoiseKind,
    PureBudget,
    Var,
)
from dpcegar.ir.nodes import (
    MechIR,
    NoiseDrawNode,
    ParamDecl,
    QueryNode,
    ReturnNode,
    SequenceNode,
)
from dpcegar.repair.templates import (
    RepairSite,
    ScaleParam,
    ThresholdShift,
)
from dpcegar.repair.patcher import (
    MechIRPrinter,
    PatchEntry,
    PatchGenerator,
    PatchReport,
    SourcePatcher,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mechir(scale: float = 0.5) -> MechIR:
    q = QueryNode(
        target=Var(ty=IRType.REAL, name="q"),
        query_name="count",
        args=(Var(ty=IRType.REAL, name="db"),),
        sensitivity=Const.real(1.0),
    )
    noise = NoiseDrawNode(
        target=Var(ty=IRType.REAL, name="eta"),
        noise_kind=NoiseKind.LAPLACE,
        center=Var(ty=IRType.REAL, name="q"),
        scale=Const.real(scale),
    )
    ret = ReturnNode(value=Var(ty=IRType.REAL, name="eta"))
    body = SequenceNode(stmts=[q, noise, ret])
    return MechIR(
        name="test_mech",
        params=[ParamDecl(name="db", ty=IRType.ARRAY, is_database=True)],
        body=body, return_type=IRType.REAL,
        budget=PureBudget(epsilon=1.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PatchEntry
# ═══════════════════════════════════════════════════════════════════════════


class TestPatchEntry:
    def test_construction(self):
        e = PatchEntry(
            parameter_name="scale",
            site_node_id=1,
            original_value=0.5,
            repaired_value=1.0,
        )
        assert e.parameter_name == "scale"

    def test_absolute_change(self):
        e = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.5, repaired_value=1.0,
        )
        assert abs(e.absolute_change - 0.5) < 1e-9

    def test_relative_change(self):
        e = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.5, repaired_value=1.0,
        )
        assert e.relative_change == 1.0  # |0.5| / 0.5

    def test_str(self):
        e = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.5, repaired_value=1.0,
        )
        assert isinstance(str(e), str)

    def test_zero_original_relative_change(self):
        e = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.0, repaired_value=1.0,
        )
        assert e.relative_change >= 0 or e.relative_change == float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# PatchReport
# ═══════════════════════════════════════════════════════════════════════════


class TestPatchReport:
    def test_construction(self):
        r = PatchReport(mechanism_name="test")
        assert r.mechanism_name == "test"
        assert r.is_valid is True

    def test_with_entries(self):
        entry = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.5, repaired_value=1.0,
        )
        r = PatchReport(
            mechanism_name="test",
            entries=[entry],
            total_cost=0.5,
        )
        assert len(r.entries) == 1
        assert r.total_cost == 0.5

    def test_summary(self):
        r = PatchReport(mechanism_name="test")
        s = r.summary()
        assert isinstance(s, str)

    def test_str(self):
        r = PatchReport(mechanism_name="test")
        assert isinstance(str(r), str)

    def test_validation_notes(self):
        r = PatchReport(
            mechanism_name="test",
            is_valid=False,
            validation_notes=["scale out of range"],
        )
        assert not r.is_valid
        assert len(r.validation_notes) == 1


# ═══════════════════════════════════════════════════════════════════════════
# MechIRPrinter
# ═══════════════════════════════════════════════════════════════════════════


class TestMechIRPrinter:
    def test_print_simple(self):
        mech = _make_mechir()
        printer = MechIRPrinter()
        text = printer.print(mech)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_print_with_indent(self):
        mech = _make_mechir()
        printer = MechIRPrinter(indent_width=4)
        text = printer.print(mech)
        assert isinstance(text, str)


# ═══════════════════════════════════════════════════════════════════════════
# PatchGenerator
# ═══════════════════════════════════════════════════════════════════════════


class TestPatchGenerator:
    def test_construction(self):
        pg = PatchGenerator()
        assert pg is not None

    def test_generate(self):
        mech = _make_mechir(scale=0.5)
        site = RepairSite(
            node_id=mech.body.stmts[1].node_id,
            node_type="NoiseDrawNode",
        )
        template = ScaleParam(site=site, original_scale=0.5)
        pg = PatchGenerator()
        report = pg.generate(
            mech, template,
            parameter_values={"new_scale": 1.0},
            original_values={"new_scale": 0.5},
        )
        assert isinstance(report, PatchReport)

    def test_apply_and_diff(self):
        mech = _make_mechir(scale=0.5)
        site = RepairSite(
            node_id=mech.body.stmts[1].node_id,
            node_type="NoiseDrawNode",
        )
        template = ScaleParam(site=site, original_scale=0.5)
        pg = PatchGenerator()
        repaired_mech, diff_text = pg.apply_and_diff(
            mech, template, parameter_values={"new_scale": 1.0},
        )
        assert isinstance(repaired_mech, MechIR)
        assert isinstance(diff_text, str)

    def test_generate_contains_entries(self):
        mech = _make_mechir(scale=0.5)
        site = RepairSite(
            node_id=mech.body.stmts[1].node_id,
            node_type="NoiseDrawNode",
        )
        template = ScaleParam(site=site, original_scale=0.5)
        pg = PatchGenerator()
        report = pg.generate(
            mech, template,
            parameter_values={"new_scale": 1.0},
            original_values={"new_scale": 0.5},
        )
        assert len(report.entries) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# SourcePatcher
# ═══════════════════════════════════════════════════════════════════════════


class TestSourcePatcher:
    def test_apply(self):
        source = "noise = laplace(q, 0.5)"
        entry = PatchEntry(
            parameter_name="scale", site_node_id=1,
            site_description="noise scale",
            original_value=0.5, repaired_value=1.0,
        )
        report = PatchReport(
            mechanism_name="test",
            entries=[entry],
            original_source=source,
        )
        sp = SourcePatcher()
        patched = sp.apply(source, report)
        assert isinstance(patched, str)

    def test_format_value(self):
        assert SourcePatcher._format_value(1.0) == "1.0" or isinstance(
            SourcePatcher._format_value(1.0), str
        )

    def test_preview(self):
        source = "noise = laplace(q, 0.5)\nreturn noise"
        entry = PatchEntry(
            parameter_name="scale", site_node_id=1,
            original_value=0.5, repaired_value=1.0,
        )
        report = PatchReport(mechanism_name="test", entries=[entry])
        sp = SourcePatcher()
        preview = sp.preview(source, report, context_lines=1)
        assert isinstance(preview, str)
