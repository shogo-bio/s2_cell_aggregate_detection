"""``intensity_ratio`` population assignment: absolute floors, then the
second/first ratio, with a third class for cells lit in both channels.

Built for the 2026-09-13 real-data situation: Cirl-GFP cells have
red/green < 0.03, Cirl-mCherry cells leak into the green channel at ~1x
(so both channels are lit and the ratio is ~1), and a large group carries
both signals in the same place (ratio 0.03-0.75), which must be reported as
its own class rather than forced into either population.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from s2_adhesion.config import PopulationAssignmentConfig, load_config
from s2_adhesion.contracts import ChannelBinding, ChannelRole, ContactEstimator, ContactRecord
from s2_adhesion.errors import ConfigError
from s2_adhesion.metrics.populations import (
    assign_populations_by_intensity_ratio,
    compute_mixing,
    label_contacts_by_population,
)


def _bindings():
    return [
        ChannelBinding("green", 0, frozenset({ChannelRole.SIGNAL}), population="GFP", color="green"),
        ChannelBinding("red", 1, frozenset({ChannelRole.SIGNAL}), population="mCherry", color="red"),
    ]


def _scene():
    """Five cells on a background of green 20 / red 0:

    1 pure GFP (green 600, red 3)          -> GFP
    2 mCherry with 1x green leak (500/500) -> mCherry (ratio 1.0)
    3 both signals (green 600, red 150)    -> double (ratio 0.25)
    4 dim autofluorescent (green 60, red 5)-> unassigned
    5 red-only (green 25, red 400)         -> mCherry (only red lit)
    """
    labels = np.zeros((3, 10, 50), np.uint32)
    green = np.full(labels.shape, 20.0, np.float32)
    red = np.zeros(labels.shape, np.float32)
    specs = {1: (600, 3), 2: (500, 500), 3: (600, 150), 4: (60, 5), 5: (25, 400)}
    for cid, (g, r) in specs.items():
        sl = (slice(None), slice(2, 8), slice(10 * (cid - 1) + 2, 10 * (cid - 1) + 8))
        labels[sl] = cid
        green[sl] = 20 + g
        red[sl] = r
    return labels, {"green": green, "red": red}


FLOORS = {"green": 150.0, "red": 100.0}


class TestAssignment:
    def test_the_five_reference_cells(self):
        labels, ch = _scene()
        rows = assign_populations_by_intensity_ratio(
            labels, ch, _bindings(), floors=FLOORS, ratio_low=0.03, ratio_high=0.75
        )
        assert {cid: r["population"] for cid, r in rows.items()} == {
            1: "GFP", 2: "mCherry", 3: "double_signal", 4: "unassigned", 5: "mCherry"
        }
        assert rows[4]["population_qc"] == "below_intensity_floor"
        assert rows[3]["population_qc"] == "both_lit_ratio_between"
        assert rows[2]["population_qc"] == "both_lit_ratio_high"
        assert rows[5]["population_qc"] is None

    def test_rows_carry_intensities_and_ratio(self):
        labels, ch = _scene()
        rows = assign_populations_by_intensity_ratio(
            labels, ch, _bindings(), floors=FLOORS, ratio_low=0.03, ratio_high=0.75
        )
        assert rows[1]["pop_intensity.green"] == pytest.approx(600.0)
        assert rows[1]["pop_intensity.red"] == pytest.approx(3.0)
        assert rows[3]["pop_ratio"] == pytest.approx(0.25)
        assert rows[2]["pop_ratio"] == pytest.approx(1.0)

    def test_floor_takes_precedence_over_ratio(self):
        """green 600, red 60: the ratio (0.1) is in the 'double' band but red
        is below its floor, so the cell is GFP, not double."""
        labels = np.zeros((2, 8, 8), np.uint32); labels[:, 1:7, 1:7] = 1
        green = np.full(labels.shape, 20.0, np.float32); red = np.zeros(labels.shape, np.float32)
        green[labels == 1] = 620; red[labels == 1] = 60
        rows = assign_populations_by_intensity_ratio(
            labels, {"green": green, "red": red}, _bindings(), floors=FLOORS,
            ratio_low=0.03, ratio_high=0.75,
        )
        assert rows[1]["population"] == "GFP"

    def test_custom_double_label_and_ratio_band(self):
        labels, ch = _scene()
        rows = assign_populations_by_intensity_ratio(
            labels, ch, _bindings(), floors=FLOORS, ratio_low=0.3, ratio_high=0.9,
            double_label="both",
        )
        # cell 3 has ratio 0.25 <= 0.3 -> first population now
        assert rows[3]["population"] == "GFP"
        # cell 2 has ratio 1.0 >= 0.9 -> still mCherry
        assert rows[2]["population"] == "mCherry"

    def test_requires_exactly_two_populations(self):
        labels, ch = _scene()
        one = [_bindings()[0]]
        with pytest.raises(ValueError, match="exactly two populations"):
            assign_populations_by_intensity_ratio(
                labels, {"green": ch["green"]}, one, floors=FLOORS, ratio_low=0.03, ratio_high=0.75
            )


class TestMixingExcludesTheThirdClass:
    @staticmethod
    def _contact(a, b):
        return ContactRecord(
            dataset_id="d", field_id="f", segmentation_run_id="r", cell_id_a=a, cell_id_b=b,
            contact_area_um2=5.0, estimator=ContactEstimator.MARCHING_CUBES,
            qualifies_as_contact=True, valid_for_contact_metrics=True,
        )

    def test_double_signal_edges_do_not_count(self):
        pops = {1: "GFP", 2: "mCherry", 3: "double_signal", 4: "GFP"}
        contacts = [self._contact(1, 2), self._contact(1, 3), self._contact(2, 3), self._contact(1, 4)]
        with_excl = compute_mixing(contacts, pops, non_population_labels=frozenset({"double_signal"}))
        assert with_excl["n_qualifying_contacts"] == 2
        assert with_excl["n_heterotypic_contacts"] == 1
        without = compute_mixing(contacts, pops)
        assert without["n_qualifying_contacts"] == 4  # unknown label counted as a population

    def test_contact_labeller_marks_third_class_as_unknown(self):
        pops = {1: "GFP", 3: "double_signal"}
        out = label_contacts_by_population(
            [self._contact(1, 3)], pops, non_population_labels=frozenset({"double_signal"})
        )
        assert out[(1, 3)]["is_heterotypic"] is None


class TestConfig:
    BASE = """
    schema_version: s2-pipeline-config/v1
    channels:
      - {channel_id: green, source_index: 0, roles: [signal], population: GFP}
      - {channel_id: red, source_index: 1, roles: [signal], population: mCherry}
    segmentation:
      strategy: direct_cellpose
      input_channel_ids: [green, red]
      channel_combination: max
    """

    @staticmethod
    def _write(tmp_path: Path, text: str, extra: str = "") -> Path:
        p = tmp_path / "cfg.yaml"
        p.write_text(textwrap.dedent(text) + textwrap.dedent(extra), encoding="utf-8")
        return p

    def test_default_is_background_mad(self, tmp_path):
        cfg = load_config(self._write(tmp_path, self.BASE))
        pop = cfg.measurement.population
        assert pop.method == "background_mad"
        assert pop.min_score_mad == 3.0 and pop.dominance_ratio == 1.5

    def test_intensity_ratio_loads_with_per_field_override(self, tmp_path):
        cfg = load_config(self._write(tmp_path, self.BASE, """
        measurement:
          population:
            method: intensity_ratio
            min_intensity_above_background: {green: 150, red: 100}
            ratio_low: 0.03
            ratio_high: 0.75
            per_field:
              field007: {green: 200}
        """))
        pop = cfg.measurement.population
        assert pop.floors_for("field000") == {"green": 150, "red": 100}
        assert pop.floors_for("field007") == {"green": 200, "red": 100}

    def test_intensity_ratio_needs_a_floor_for_every_population_channel(self, tmp_path):
        with pytest.raises(ConfigError, match="missing \\['red'\\]"):
            load_config(self._write(tmp_path, self.BASE, """
            measurement:
              population:
                method: intensity_ratio
                min_intensity_above_background: {green: 150}
            """))

    def test_floor_for_an_unknown_channel_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="'blue'"):
            load_config(self._write(tmp_path, self.BASE, """
            measurement:
              population:
                min_intensity_above_background: {blue: 150}
            """))

    def test_ratio_band_must_be_ordered(self, tmp_path):
        with pytest.raises(ConfigError, match="ratio_low < ratio_high"):
            load_config(self._write(tmp_path, self.BASE, """
            measurement:
              population:
                ratio_low: 0.9
                ratio_high: 0.5
            """))

    def test_double_label_cannot_collide_with_a_population(self, tmp_path):
        with pytest.raises(ConfigError, match="collides"):
            load_config(self._write(tmp_path, self.BASE, """
            measurement:
              population:
                double_label: GFP
            """))

    def test_unknown_method_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="method"):
            load_config(self._write(tmp_path, self.BASE, """
            measurement:
              population:
                method: kmeans
            """))

    def test_shipped_both_populations_config_uses_intensity_ratio(self):
        cfg = load_config(Path("configs/cirl_gfp_vs_cirl_m_both.yaml"))
        pop = cfg.measurement.population
        assert pop.method == "intensity_ratio"
        assert pop.min_intensity_above_background == {"green": 150, "red": 100}
        assert cfg.segmentation.min_z_extent_planes == 3


def test_population_config_dataclass_floors_helper():
    pop = PopulationAssignmentConfig(
        min_intensity_above_background={"green": 1.0}, per_field={"f": {"green": 2.0}}
    )
    assert pop.floors_for("f") == {"green": 2.0}
    assert pop.floors_for("g") == {"green": 1.0}
