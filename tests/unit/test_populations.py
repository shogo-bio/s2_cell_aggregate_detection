"""Population assignment and the adhesion mixing index.

The mixing index is validated against layouts whose answer is known by
construction: a checkerboard (every contact heterotypic) must read as mixing,
two separated blocks (every contact homotypic) as segregation, and a random
arrangement as ~1.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2_adhesion.contracts import ChannelBinding, ChannelRole, ContactRecord, ContactEstimator
from s2_adhesion.metrics.populations import (
    assign_populations,
    compute_mixing,
    label_contacts_by_population,
)


def _bindings():
    return [
        ChannelBinding("green", 0, frozenset({ChannelRole.SIGNAL}),
                       population="popA", color="green"),
        ChannelBinding("red", 1, frozenset({ChannelRole.SIGNAL}),
                       population="popB", color="red"),
    ]


def _two_cells_different_brightness():
    """A green cell and a red cell, with FITC far brighter than mCherry.

    Cell 1 is a green cell (bright green, dark red); cell 2 is a red cell (dim
    red, dark green). The green channel is scaled up 8x to mimic FITC >> mCherry.
    A naive 'green > red' test would call BOTH cells green.
    """
    labels = np.zeros((4, 20, 40), np.uint32)
    labels[:, 4:16, 4:16] = 1
    labels[:, 4:16, 24:36] = 2

    green = np.full(labels.shape, 5.0, np.float32)   # background
    red = np.full(labels.shape, 5.0, np.float32)
    green[labels == 1] = 800.0                       # bright green cell
    red[labels == 1] = 6.0
    green[labels == 2] = 7.0
    red[labels == 2] = 60.0                          # dim (but real) red cell
    return labels, {"green": green, "red": red}


class TestAssignment:
    def test_assigns_despite_channel_brightness_difference(self):
        labels, channels = _two_cells_different_brightness()
        out = assign_populations(labels, channels, _bindings())
        assert out[1]["population"] == "popA"
        assert out[2]["population"] == "popB", (
            "the dim red cell was misread as green -- per-channel normalisation "
            "is not working"
        )

    def test_a_dim_cell_is_left_unassigned_not_forced(self):
        labels = np.zeros((4, 20, 20), np.uint32)
        labels[:, 4:16, 4:16] = 1
        green = np.full(labels.shape, 5.0, np.float32)
        red = np.full(labels.shape, 5.0, np.float32)
        green[labels == 1] = 6.0  # barely above background
        out = assign_populations(labels, green_red(green, red), _bindings())
        assert out[1]["population"] == "unassigned"
        assert out[1]["population_qc"] == "below_min_score"

    def test_a_cell_bright_in_both_is_ambiguous(self):
        """Real for co-expression or bleed-through; must not be silently picked."""
        labels = np.zeros((4, 20, 20), np.uint32)
        labels[:, 4:16, 4:16] = 1
        green = np.full(labels.shape, 5.0, np.float32)
        red = np.full(labels.shape, 5.0, np.float32)
        green[labels == 1] = 400.0
        red[labels == 1] = 400.0
        out = assign_populations(labels, {"green": green, "red": red}, _bindings())
        assert out[1]["population"] == "ambiguous"
        assert out[1]["population_qc"] == "two_populations_present"

    def test_several_channels_can_share_one_population(self):
        """Co-transfected markers of the same group: brightest one wins for the group."""
        bindings = [
            ChannelBinding("g1", 0, frozenset({ChannelRole.SIGNAL}), population="popA"),
            ChannelBinding("g2", 1, frozenset({ChannelRole.SIGNAL}), population="popA"),
            ChannelBinding("b", 2, frozenset({ChannelRole.SIGNAL}), population="popB"),
        ]
        labels = np.zeros((4, 20, 20), np.uint32)
        labels[:, 4:16, 4:16] = 1
        g1 = np.full(labels.shape, 5.0, np.float32)
        g2 = np.full(labels.shape, 5.0, np.float32)
        b = np.full(labels.shape, 5.0, np.float32)
        g1[labels == 1] = 6.0     # this group-A marker is dim
        g2[labels == 1] = 300.0   # this one is bright
        out = assign_populations(labels, {"g1": g1, "g2": g2, "b": b}, bindings)
        assert out[1]["population"] == "popA"


def green_red(green, red):
    return {"green": green, "red": red}


def _contact(a, b, qualifies=True, valid=True):
    lo, hi = sorted((a, b))
    return ContactRecord(
        dataset_id="d", field_id="f", segmentation_run_id="r",
        cell_id_a=lo, cell_id_b=hi,
        contact_area_um2=10.0, estimator=ContactEstimator.MARCHING_CUBES,
        qualifies_as_contact=qualifies, valid_for_contact_metrics=valid,
    )


class TestMixingIndex:
    def test_checkerboard_reads_as_mixing(self):
        """Alternating populations -> every contact heterotypic -> index > 1."""
        pop = {i: ("popA" if i % 2 == 0 else "popB") for i in range(1, 11)}
        contacts = [_contact(i, i + 1) for i in range(1, 10)]  # chain 1-2-...-10
        m = compute_mixing(contacts, pop)
        assert m["n_heterotypic_contacts"] == 9
        assert m["n_homotypic_contacts"] == 0
        assert m["mixing_index"] > 1.0

    def test_two_blocks_read_as_segregation(self):
        """Each population clumps with itself -> every contact homotypic -> index < 1."""
        pop = {1: "popA", 2: "popA", 3: "popA", 4: "popB", 5: "popB", 6: "popB"}
        contacts = [_contact(1, 2), _contact(2, 3), _contact(4, 5), _contact(5, 6)]
        m = compute_mixing(contacts, pop)
        assert m["n_heterotypic_contacts"] == 0
        assert m["mixing_index"] < 1.0

    def test_single_population_has_no_mixing_index(self):
        pop = {1: "popA", 2: "popA", 3: "popA"}
        m = compute_mixing([_contact(1, 2), _contact(2, 3)], pop)
        assert m["mixing_index"] is None
        assert m["mixing_qc"] == "single_population"

    def test_ambiguous_and_unassigned_cells_are_excluded(self):
        pop = {1: "popA", 2: "popB", 3: "ambiguous", 4: "unassigned"}
        contacts = [_contact(1, 2), _contact(1, 3), _contact(2, 4)]
        m = compute_mixing(contacts, pop)
        # Only the 1-2 edge counts.
        assert m["n_qualifying_contacts"] == 1
        assert m["n_heterotypic_contacts"] == 1

    def test_non_qualifying_and_invalid_contacts_are_excluded(self):
        pop = {1: "popA", 2: "popB", 3: "popA"}
        contacts = [
            _contact(1, 2, qualifies=False),
            _contact(2, 3, valid=False),
        ]
        m = compute_mixing(contacts, pop)
        assert m["n_qualifying_contacts"] == 0
        assert m["mixing_index"] is None

    def test_per_pair_counts_are_reported(self):
        pop = {1: "popA", 2: "popB", 3: "popB"}
        contacts = [_contact(1, 2), _contact(2, 3)]
        m = compute_mixing(contacts, pop)
        assert m["contacts.popA__popB"] == 1
        assert m["contacts.popB__popB"] == 1


class TestContactLabelling:
    def test_labels_heterotypic_and_homotypic(self):
        pop = {1: "popA", 2: "popB", 3: "popA"}
        labelled = label_contacts_by_population([_contact(1, 2), _contact(1, 3)], pop)
        assert labelled[(1, 2)]["is_heterotypic"] is True
        assert labelled[(1, 3)]["is_heterotypic"] is False

    def test_unknown_population_gives_none_not_a_guess(self):
        pop = {1: "popA", 2: "unassigned"}
        labelled = label_contacts_by_population([_contact(1, 2)], pop)
        assert labelled[(1, 2)]["is_heterotypic"] is None


def test_no_ml_imports():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, s2_adhesion.metrics.populations as m; "
            "leaked=[k for k in sys.modules if k.split('.')[0] in ('torch','cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
