"""Assign each cell to a population, and measure how the populations mix.

This is the readout of a cell-aggregation adhesion assay. Two (or more) cell
populations are each marked by a different fluorescent channel and mixed. If the
molecule under test mediates adhesion, cells of the relevant populations end up
touching more than chance would predict -- they mix; otherwise they segregate.

Two things make this non-trivial and are handled explicitly here:

1. Populations are told apart by WHICH channel a cell is bright in, but channel
   brightness is not comparable across channels -- FITC routinely outshines
   mCherry several-fold, and a raw "green > red" test then labels almost
   everything green. Every channel is therefore scored RELATIVE TO ITS OWN
   background before cells are assigned.

2. A channel maps to a population via config (``ChannelBinding.population``), and
   several channels may share one population (two co-transfected markers of the
   same group). A cell's population is the group whose channels it is brightest
   in, not the single brightest channel.

Nothing here reconstructs a surface, so it is unaffected by the coarse axial
sampling that makes contact AREA unreliable. Population identity and
mixing are counting operations, robust at this resolution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..contracts import ChannelBinding, ContactRecord, Scalar

UNASSIGNED = "unassigned"
AMBIGUOUS = "ambiguous"
# Labels that are not a population: cells carrying them never enter the
# mixing index or the homotypic/heterotypic contact counts. The configured
# ``double_label`` is added at call time.
_NON_POPULATION_BASE = frozenset({UNASSIGNED, AMBIGUOUS})


def _is_population(label: str | None, extra_non_population: frozenset[str]) -> bool:
    return label is not None and label not in _NON_POPULATION_BASE and label not in extra_non_population


def _median_above_background(
    intensity: NDArray[np.floating], mask: NDArray[np.bool_], outside: NDArray[np.bool_]
) -> float:
    """Cell median minus the channel's background median (counts)."""
    if not mask.any():
        return 0.0
    bg = intensity[outside]
    background = float(np.median(bg)) if bg.size else 0.0
    return float(np.median(intensity[mask])) - background


def assign_populations_by_intensity_ratio(
    labels: NDArray[np.uint32],
    channels: Mapping[str, NDArray[np.floating]],
    channel_bindings: Sequence[ChannelBinding],
    *,
    floors: Mapping[str, float],
    ratio_low: float,
    ratio_high: float,
    double_label: str = "double_signal",
) -> dict[int, dict[str, Scalar]]:
    """Two-population call from absolute floors and the second/first ratio.

    See ``config.PopulationAssignmentConfig`` for the rule. Populations are
    ordered by the first channel binding that names each of them, so with
    ``green`` bound before ``red`` the ratio is red/green.

    Per cell the row carries ``pop_intensity.<channel>`` (median above
    background, counts), ``pop_ratio`` (second/first intensity, ``None``
    when the first is <= 0), ``population`` and ``population_qc``.
    """
    ordered_pops: list[str] = []
    pop_channels: dict[str, list[str]] = defaultdict(list)
    for binding in channel_bindings:
        if binding.population and binding.channel_id in channels:
            if binding.population not in ordered_pops:
                ordered_pops.append(binding.population)
            pop_channels[binding.population].append(binding.channel_id)
    if len(ordered_pops) != 2:
        raise ValueError(
            f"intensity_ratio assignment needs exactly two populations, got {ordered_pops}"
        )
    first, second = ordered_pops
    outside = labels == 0
    out: dict[int, dict[str, Scalar]] = {}
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids > 0]

    for cell_id in cell_ids:
        mask = labels == cell_id
        intensity = {
            cid: _median_above_background(channels[cid], mask, outside)
            for cids in pop_channels.values() for cid in cids
        }
        row: dict[str, Scalar] = {
            f"pop_intensity.{cid}": round(v, 2) for cid, v in intensity.items()
        }
        # A population's intensity is its brightest channel, in units of that
        # channel's floor; "lit" means >= 1.
        def best(pop: str) -> tuple[float, float]:
            scores = [(intensity[cid] / floors[cid] if floors[cid] > 0 else float("inf"),
                       intensity[cid]) for cid in pop_channels[pop]]
            return max(scores)
        (s1, i1), (s2, i2) = best(first), best(second)
        lit1, lit2 = s1 >= 1.0, s2 >= 1.0
        ratio = (i2 / i1) if i1 > 0 else None
        row["pop_ratio"] = None if ratio is None else round(ratio, 4)
        if not lit1 and not lit2:
            row["population"] = UNASSIGNED
            row["population_qc"] = "below_intensity_floor"
        elif lit1 and not lit2:
            row["population"] = first
            row["population_qc"] = None
        elif lit2 and not lit1:
            row["population"] = second
            row["population_qc"] = None
        elif ratio is not None and ratio <= ratio_low:
            row["population"] = first
            row["population_qc"] = "both_lit_ratio_low"
        elif ratio is not None and ratio >= ratio_high:
            row["population"] = second
            row["population_qc"] = "both_lit_ratio_high"
        else:
            row["population"] = double_label
            row["population_qc"] = "both_lit_ratio_between"
        out[int(cell_id)] = row
    return out


def _background_relative_score(
    intensity: NDArray[np.floating], mask: NDArray[np.bool_], outside: NDArray[np.bool_]
) -> float:
    """Mean signal inside a cell, in units of the channel's own background MAD.

    Expressing every channel on its own noise scale is what makes a green
    channel and a red channel comparable despite very different absolute
    brightness.
    """
    if not mask.any():
        return 0.0
    bg = intensity[outside]
    if bg.size == 0:
        median, mad = 0.0, 1.0
    else:
        median = float(np.median(bg))
        mad = float(np.median(np.abs(bg - median))) * 1.4826  # ~std for normal noise
        if mad <= 0:
            mad = 1.0
    return (float(intensity[mask].mean()) - median) / mad


def assign_populations(
    labels: NDArray[np.uint32],
    channels: Mapping[str, NDArray[np.floating]],
    channel_bindings: Sequence[ChannelBinding],
    *,
    min_score: float = 3.0,
    dominance_ratio: float = 1.5,
) -> dict[int, dict[str, Scalar]]:
    """Assign each cell to the population it is brightest in.

    ``min_score``: a cell must exceed this many background-MADs in a population's
    channels to be assigned to it, so a dim cell is not forced into a group.

    ``dominance_ratio``: the best population's score must beat the runner-up by
    this factor, otherwise the cell is ``"ambiguous"`` -- it reads as belonging
    to two groups (real for co-expression, or bleed-through/overlap). Ambiguous
    and unassigned are distinct: one is "in two", the other "in none".
    """
    # Which channels vote for which population.
    pop_channels: dict[str, list[str]] = defaultdict(list)
    for binding in channel_bindings:
        if binding.population and binding.channel_id in channels:
            pop_channels[binding.population].append(binding.channel_id)

    outside = labels == 0
    out: dict[int, dict[str, Scalar]] = {}

    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids > 0]

    for cell_id in cell_ids:
        mask = labels == cell_id
        pop_scores: dict[str, float] = {}
        for pop, cids in pop_channels.items():
            # A population's score is the strongest of its channels -- a cell
            # co-transfected with two markers of one group need only show one.
            pop_scores[pop] = max(
                _background_relative_score(channels[cid], mask, outside) for cid in cids
            )

        row: dict[str, Scalar] = {
            f"pop_score.{pop}": round(score, 3) for pop, score in pop_scores.items()
        }

        if not pop_scores:
            row["population"] = UNASSIGNED
            row["population_qc"] = "no_population_channels_configured"
            out[int(cell_id)] = row
            continue

        ranked = sorted(pop_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_pop, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < min_score:
            row["population"] = UNASSIGNED
            row["population_qc"] = "below_min_score"
        elif second_score >= min_score and best_score < dominance_ratio * max(
            second_score, 1e-9
        ):
            row["population"] = "ambiguous"
            row["population_qc"] = "two_populations_present"
        else:
            row["population"] = best_pop
            row["population_qc"] = None

        out[int(cell_id)] = row

    return out


def compute_mixing(
    contacts: Sequence[ContactRecord],
    cell_population: Mapping[int, str],
    *,
    non_population_labels: frozenset[str] = frozenset(),
) -> dict[str, Scalar]:
    """Field-level mixing statistics from the qualifying contact graph.

    The core adhesion readout. Counts contacts by the population pair they join
    (homotypic A-A, heterotypic A-B, ...) and compares the heterotypic fraction
    against what random mixing of the same populations would give.

    ``mixing_index`` is observed heterotypic fraction / expected-under-random.
    > 1 means the populations touch across groups MORE than chance (they mix,
    i.e. cross-population adhesion); < 1 means they segregate. It is undefined
    (``None``) when only one population is present or there are no qualifying
    contacts, rather than being reported as a spurious 0 or 1.
    """
    edges: list[tuple[str, str]] = []
    for c in contacts:
        if not (c.qualifies_as_contact and c.valid_for_contact_metrics):
            continue
        pa = cell_population.get(c.cell_id_a)
        pb = cell_population.get(c.cell_id_b)
        if not (_is_population(pa, non_population_labels)
                and _is_population(pb, non_population_labels)):
            continue
        edges.append(tuple(sorted((pa, pb))))

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for e in edges:
        pair_counts[e] += 1

    total = len(edges)
    homotypic = sum(n for (a, b), n in pair_counts.items() if a == b)
    heterotypic = total - homotypic

    result: dict[str, Scalar] = {
        "n_qualifying_contacts": total,
        "n_homotypic_contacts": homotypic,
        "n_heterotypic_contacts": heterotypic,
    }
    for (a, b), n in sorted(pair_counts.items()):
        result[f"contacts.{a}__{b}"] = n

    # Expected heterotypic fraction under random pairing of the assigned cells.
    pop_of_edge_endpoints: dict[str, int] = defaultdict(int)
    for (a, b), n in pair_counts.items():
        pop_of_edge_endpoints[a] += n
        pop_of_edge_endpoints[b] += n
    populations = list(pop_of_edge_endpoints)

    if total == 0 or len(populations) < 2:
        result["heterotypic_fraction"] = None
        result["mixing_index"] = None
        result["mixing_qc"] = (
            "single_population" if len(populations) < 2 else "no_contacts"
        )
        return result

    endpoints = 2 * total
    frac = {p: pop_of_edge_endpoints[p] / endpoints for p in populations}
    # P(random edge is heterotypic) = 1 - sum p_i^2
    expected_hetero = 1.0 - sum(f * f for f in frac.values())
    observed_hetero = heterotypic / total

    result["heterotypic_fraction"] = round(observed_hetero, 4)
    result["expected_heterotypic_fraction"] = round(expected_hetero, 4)
    result["mixing_index"] = (
        round(observed_hetero / expected_hetero, 4) if expected_hetero > 0 else None
    )
    result["mixing_qc"] = None
    return result


def label_contacts_by_population(
    contacts: Sequence[ContactRecord],
    cell_population: Mapping[int, str],
    *,
    non_population_labels: frozenset[str] = frozenset(),
) -> dict[tuple[int, int], dict[str, Scalar]]:
    """Tag each contact with the populations it joins and whether it is heterotypic."""
    out: dict[tuple[int, int], dict[str, Scalar]] = {}
    for c in contacts:
        pa = cell_population.get(c.cell_id_a, UNASSIGNED)
        pb = cell_population.get(c.cell_id_b, UNASSIGNED)
        known = _is_population(pa, non_population_labels) and _is_population(
            pb, non_population_labels
        )
        out[(c.cell_id_a, c.cell_id_b)] = {
            "population_a": pa,
            "population_b": pb,
            "is_heterotypic": (pa != pb) if known else None,
        }
    return out
