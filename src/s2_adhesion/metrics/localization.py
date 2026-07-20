"""Sub-cellular localization: signed-distance profiles, shell enrichment scores,
per-cell colocalization, and an optional categorical call.

Pure numpy/scipy -- no ML imports, so this module works with no ML package
installed at all (mirrors ``metrics.qc``, ``metrics.geometry``, ``metrics.aggregates``).

SCIENTIFIC CONSTRAINT THIS MODULE MUST RESPECT EVERYWHERE: confocal axial
resolution is roughly 0.5-0.8 um; a membrane is roughly 5 nm thick. No
measurement in this file can determine which side of a membrane a molecule
sits on -- that is three orders of magnitude below what the optics resolve.
Every score here is a resolution-limited ENRICHMENT measurement (a ratio of
mean intensities between two regions of the same cell), never a statement of
molecular sidedness. This is also why there is no categorical "membrane"
class: ``membrane_enrichment_score`` is reported as a continuous ratio only.
The categorical classifier (``classify_localization``) additionally defaults
to disabled (``LocalizationDecisionConfig.enabled = False``) and must return
``"indeterminate"`` for everything while it is disabled -- categorical calls
need thresholds calibrated against real biological controls that nobody has
produced yet, while the continuous profiles and scores need no calibration
and are always available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from ..config import LocalizationConfig, LocalizationDecisionConfig, ThresholdSpec
from ..contracts import LocalizationProfileRecord, Scalar, VoxelGeometry
from ..errors import MeasurementError

# 1.4826 converts a median absolute deviation into an equivalent standard
# deviation for normally-distributed noise -- the standard robust-statistics
# scale factor -- so a ``mad_multiplier`` in ``ThresholdSpec`` reads on the
# same footing a sigma-multiplier would for Gaussian background.
_MAD_TO_SIGMA = 1.4826

_INDETERMINATE = "indeterminate"
_MIXED = "mixed"


def _cell_ids(cells: NDArray[np.uint32]) -> list[int]:
    ids = np.unique(cells)
    return sorted(int(i) for i in ids if i != 0)


def _validate_edges(edges: tuple[float, ...]) -> NDArray[np.float64]:
    arr = np.asarray(edges, dtype=np.float64)
    if arr.size < 2:
        raise MeasurementError("signed_distance_bin_edges_um needs at least two edges")
    if np.any(np.diff(arr) <= 0):
        raise MeasurementError(
            f"signed_distance_bin_edges_um must strictly increase: {edges}"
        )
    return arr


def _bin_index(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Which ``[edges[i], edges[i+1])`` bin each value falls in; -1 if out of range.

    The final bin is closed on the right so a value exactly at the outermost
    edge is still counted rather than silently dropped.
    """
    n_bins = edges.size - 1
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.where(values == edges[-1], n_bins - 1, idx)
    idx = np.where((values < edges[0]) | (values > edges[-1]), -1, idx)
    return idx


def _per_cell_signed_distances(
    cells: NDArray[np.uint32], spacing: tuple[float, float, float]
) -> tuple[NDArray[np.uint32], dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    """Nearest-cell assignment plus per-cell ``(d_in, d_out)`` distance fields.

    ``d_in = distance_transform_edt(cells == cid, sampling=spacing)`` is the
    positive inside-depth (zero outside that cell). ``d_out =
    distance_transform_edt(cells != cid, sampling=spacing)`` is the positive
    outside-distance to that cell (zero inside it); this is exactly the
    per-cell distance field the nearest-cell assignment needs, so both are
    computed together.

    Every background voxel (``cells == 0``) is assigned to exactly ONE
    nearest cell id, ties broken by the lower id: cell ids are processed in
    ascending order and a voxel's assignment is only overwritten on a STRICT
    distance improvement, so the first (lowest-id) cell to achieve the
    minimum distance keeps it. Foreground voxels keep whichever id
    momentarily has the smallest ``d_out`` there, which is irrelevant since
    callers only consult ``nearest_cell`` for background voxels.
    """
    ids = _cell_ids(cells)
    best_dist = np.full(cells.shape, np.inf, dtype=np.float64)
    nearest = np.zeros(cells.shape, dtype=np.uint32)
    per_cell: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    spacing_arr = tuple(float(s) for s in spacing)
    for cid in ids:
        mask = cells == cid
        d_in = distance_transform_edt(mask, sampling=spacing_arr)
        d_out = distance_transform_edt(~mask, sampling=spacing_arr)
        per_cell[cid] = (d_in, d_out)
        take = d_out < best_dist
        best_dist = np.where(take, d_out, best_dist)
        nearest = np.where(take, np.uint32(cid), nearest)
    return nearest, per_cell


def compute_signed_distance_profiles(
    cells: NDArray[np.uint32],
    geometry: VoxelGeometry,
    dataset_id: str,
    field_id: str,
    channels: Mapping[str, NDArray[np.floating]],
    config: LocalizationConfig,
    corrected_channels: Mapping[str, NDArray[np.floating]] | None = None,
) -> tuple[LocalizationProfileRecord, ...]:
    """Binned signed-distance intensity profiles, relative to the cell boundary.

    ``cells`` is a raw ``ZYX`` ``uint32`` label array (background 0).
    ``channels`` maps channel id to a raw ``ZYX`` intensity array of the same
    shape; ``corrected_channels`` is the same shape of background-subtracted
    intensity (optional -- when a channel is absent, or the mapping itself is
    omitted, ``integrated_corrected_intensity`` is ``None`` for that channel).

    Sign convention (frozen): inside a cell the distance is
    ``+distance_transform_edt(mask, sampling=spacing)``; outside it is
    ``-distance_transform_edt(~mask, sampling=spacing)``. ``sampling=spacing``
    is mandatory -- an unweighted voxel-unit EDT would be wrong by
    ``spacing[0] / min(spacing[1], spacing[2])`` in Z under anisotropic
    sampling.

    Every background voxel (not inside any cell) is assigned to exactly ONE
    nearest cell (ties broken by lower cell id, see
    ``_per_cell_signed_distances``), so summing ``voxel_count`` or
    ``integrated_corrected_intensity`` for a given bin across every cell's
    profile never double counts a voxel.

    One row is emitted per ``(cell, channel, bin)`` covering every bin in
    ``config.signed_distance_bin_edges_um``, including empty bins
    (``voxel_count=0``, intensities ``None``), so every cell's profile spans
    an identical, complete grid.
    """
    edges = _validate_edges(config.signed_distance_bin_edges_um)
    n_bins = edges.size - 1
    spacing = geometry.spacing_um_zyx
    voxel_volume = geometry.voxel_volume_um3
    corrected_channels = corrected_channels or {}

    nearest_cell, per_cell = _per_cell_signed_distances(cells, spacing)

    records: list[LocalizationProfileRecord] = []
    for cid in _cell_ids(cells):
        d_in, d_out = per_cell[cid]
        signed = d_in - d_out
        valid_mask = (cells == cid) | ((cells == 0) & (nearest_cell == cid))
        bin_idx = _bin_index(signed, edges)
        bin_idx = np.where(valid_mask, bin_idx, -1)

        for channel_id, raw in channels.items():
            corrected = corrected_channels.get(channel_id)
            for i in range(n_bins):
                bin_mask = bin_idx == i
                voxel_count = int(np.count_nonzero(bin_mask))
                sampled_volume_um3 = voxel_count * voxel_volume
                if voxel_count > 0:
                    mean_intensity = float(raw[bin_mask].astype(np.float64).mean())
                    integrated_corrected_intensity = (
                        float(corrected[bin_mask].astype(np.float64).sum()) * voxel_volume
                        if corrected is not None
                        else None
                    )
                else:
                    mean_intensity = None
                    integrated_corrected_intensity = None

                records.append(
                    LocalizationProfileRecord(
                        dataset_id=dataset_id,
                        field_id=field_id,
                        cell_id=cid,
                        channel_id=channel_id,
                        reference="cell_boundary",
                        bin_start_um=float(edges[i]),
                        bin_end_um=float(edges[i + 1]),
                        voxel_count=voxel_count,
                        sampled_volume_um3=sampled_volume_um3,
                        mean_intensity=mean_intensity,
                        integrated_corrected_intensity=integrated_corrected_intensity,
                    )
                )
    return tuple(records)


def compute_shell_enrichment_scores(
    cells: NDArray[np.uint32],
    geometry: VoxelGeometry,
    channels: Mapping[str, NDArray[np.floating]],
    config: LocalizationConfig,
    background_noise_std: Mapping[str, float] | None = None,
) -> dict[int, dict[str, Scalar]]:
    """Per-cell, per-channel scalar enrichment scores from three shells.

    Shells, all measured from the cell boundary using the same signed
    distance convention as ``compute_signed_distance_profiles``:

    * outer shell: background voxels within ``outer_shell_width_um``
      outside the boundary, restricted to this cell's nearest-cell
      assignment (never a neighbour's territory -- see
      ``_per_cell_signed_distances``).
    * inner shell: voxels inside the cell within ``inner_shell_width_um``
      of the boundary.
    * core: voxels inside the cell deeper than ``inner_shell_width_um``.

    ``eps`` (from ``background_noise_std``, per channel; 0.0 when a channel
    is absent from the mapping or the mapping is omitted) is a RECORDED
    parameter the caller scales to that channel's background noise -- never
    a hardcoded constant -- added to both numerator and denominator of every
    ratio so a near-zero mean does not blow up into a meaningless ratio.

    Returned keys, per cell id, are ``f"{channel_id}__{score_name}"`` for
    ``score_name`` in ``extracellular_enrichment_score``,
    ``intracellular_enrichment_score``, ``membrane_enrichment_score``. Any
    score is ``None`` when either shell it compares is empty, or when eps
    does not lift both sides of the ratio above zero. ``membrane_enrichment_score``
    is a continuous readout only -- see the module docstring for why there is
    no categorical membrane class.
    """
    background_noise_std = background_noise_std or {}
    spacing = geometry.spacing_um_zyx
    nearest_cell, per_cell = _per_cell_signed_distances(cells, spacing)

    out: dict[int, dict[str, Scalar]] = {cid: {} for cid in _cell_ids(cells)}
    for cid in _cell_ids(cells):
        d_in, d_out = per_cell[cid]
        cell_mask = cells == cid
        outer_mask = (
            (cells == 0)
            & (nearest_cell == cid)
            & (d_out <= config.outer_shell_width_um)
        )
        inner_mask = cell_mask & (d_in <= config.inner_shell_width_um)
        core_mask = cell_mask & (d_in > config.inner_shell_width_um)

        for channel_id, raw in channels.items():
            eps = float(background_noise_std.get(channel_id, 0.0))
            raw64 = raw.astype(np.float64)

            outer_mean = float(raw64[outer_mask].mean()) if np.any(outer_mask) else None
            inner_mean = float(raw64[inner_mask].mean()) if np.any(inner_mask) else None
            core_mean = float(raw64[core_mask].mean()) if np.any(core_mask) else None

            def _log2_ratio(numer: float | None, denom: float | None) -> float | None:
                if numer is None or denom is None:
                    return None
                n, d = numer + eps, denom + eps
                if n <= 0 or d <= 0:
                    return None
                return float(np.log2(n / d))

            row = out[cid]
            row[f"{channel_id}__extracellular_enrichment_score"] = _log2_ratio(
                outer_mean, inner_mean
            )
            row[f"{channel_id}__intracellular_enrichment_score"] = _log2_ratio(
                core_mean, outer_mean
            )
            row[f"{channel_id}__membrane_enrichment_score"] = _log2_ratio(
                inner_mean, core_mean
            )

    return out


def apply_threshold_spec(
    values: NDArray[np.floating],
    spec: ThresholdSpec,
    background_median: float = 0.0,
    background_mad: float = 0.0,
) -> NDArray[np.bool_]:
    """Binarise ``values >= threshold`` with the threshold from ``ThresholdSpec``.

    ``mode="fixed"`` uses ``spec.fixed_value`` directly. ``mode="background_mad"``
    uses ``background_median + spec.mad_multiplier * _MAD_TO_SIGMA * background_mad``,
    where ``background_median``/``background_mad`` are the caller's measured
    background statistics for this channel (this function never estimates
    them itself). There is deliberately no automatic/Otsu-style threshold
    path -- every threshold traces to an explicit, recorded ``ThresholdSpec``.
    """
    if spec.mode == "fixed":
        if spec.fixed_value is None:
            raise MeasurementError("ThresholdSpec(mode='fixed') requires fixed_value")
        threshold = spec.fixed_value
    elif spec.mode == "background_mad":
        if spec.mad_multiplier is None:
            raise MeasurementError(
                "ThresholdSpec(mode='background_mad') requires mad_multiplier"
            )
        threshold = background_median + spec.mad_multiplier * _MAD_TO_SIGMA * background_mad
    else:
        raise MeasurementError(f"unknown ThresholdSpec.mode {spec.mode!r}")
    return values >= threshold


def compute_colocalization(
    cells: NDArray[np.uint32],
    geometry: VoxelGeometry,
    channels: Mapping[str, NDArray[np.floating]],
    config: LocalizationConfig,
    background_stats: Mapping[str, tuple[float, float]] | None = None,
) -> dict[int, dict[str, Scalar]]:
    """Per-cell colocalization for every ``config.colocalization_pairs`` entry.

    Computed PER CELL, never per image -- a whole-image coefficient hides
    all cell-to-cell variation. For each pair, returned keys per cell id are
    ``f"{signal_channel_id}_vs_{reference_channel_id}__{name}"`` for ``name``
    in ``pearson_r``, ``manders_signal_in_reference``,
    ``manders_reference_in_signal``, ``organelle_surface_enrichment_score``.

    ``pearson_r`` is ``None`` on zero variance in either channel within the
    cell (a constant signal has no correlation to report, not a coincidental
    +-1). Manders coefficients use binary masks from ``apply_threshold_spec``
    on ``pair.signal_threshold`` / ``pair.reference_threshold``:
    ``manders_signal_in_reference`` is the fraction of total signal intensity
    (within the cell) that falls inside the thresholded reference region;
    ``manders_reference_in_signal`` is the symmetric quantity.

    ``organelle_surface_enrichment_score`` is computed only when
    ``pair.reference_role == "organelle_marker"`` and
    ``pair.organelle_surface_band_um`` is set: it thresholds the reference
    channel to an organelle mask (within the cell), takes a signed-distance
    band of that width around the mask's boundary, and compares the signal
    channel's mean inside the band to its mean over the rest of the cell.
    ``background_stats`` maps channel id to ``(median, mad)``, consulted by
    ``apply_threshold_spec`` for ``mode="background_mad"`` and as the eps
    source (the signal channel's mad) for the organelle ratio; a channel
    absent from the mapping defaults to ``(0.0, 0.0)``.
    """
    background_stats = background_stats or {}
    spacing = geometry.spacing_um_zyx
    cell_ids = _cell_ids(cells)
    out: dict[int, dict[str, Scalar]] = {cid: {} for cid in cell_ids}

    for pair in config.colocalization_pairs:
        prefix = f"{pair.signal_channel_id}_vs_{pair.reference_channel_id}"
        signal_full = channels[pair.signal_channel_id]
        reference_full = channels[pair.reference_channel_id]
        signal_full64 = signal_full.astype(np.float64)
        reference_full64 = reference_full.astype(np.float64)

        sig_med, sig_mad = background_stats.get(pair.signal_channel_id, (0.0, 0.0))
        ref_med, ref_mad = background_stats.get(pair.reference_channel_id, (0.0, 0.0))
        signal_mask_full = apply_threshold_spec(
            signal_full, pair.signal_threshold, sig_med, sig_mad
        )
        reference_mask_full = apply_threshold_spec(
            reference_full, pair.reference_threshold, ref_med, ref_mad
        )

        for cid in cell_ids:
            cell_mask = cells == cid
            sv = signal_full64[cell_mask]
            rv = reference_full64[cell_mask]

            pearson_r: float | None = None
            if sv.size >= 2 and np.std(sv) > 0 and np.std(rv) > 0:
                pearson_r = float(np.corrcoef(sv, rv)[0, 1])

            sig_mask_cell = signal_mask_full & cell_mask
            ref_mask_cell = reference_mask_full & cell_mask
            total_signal = float(sv.sum())
            total_reference = float(rv.sum())
            manders_signal_in_reference = (
                float(signal_full64[ref_mask_cell].sum()) / total_signal
                if total_signal != 0
                else None
            )
            manders_reference_in_signal = (
                float(reference_full64[sig_mask_cell].sum()) / total_reference
                if total_reference != 0
                else None
            )

            organelle_score: float | None = None
            if (
                pair.reference_role == "organelle_marker"
                and pair.organelle_surface_band_um is not None
            ):
                organelle_mask_cell = reference_mask_full & cell_mask
                has_pos = bool(np.any(organelle_mask_cell))
                has_neg = bool(np.any(cell_mask & ~organelle_mask_cell))
                if has_pos and has_neg:
                    d_in = distance_transform_edt(organelle_mask_cell, sampling=spacing)
                    d_out = distance_transform_edt(~organelle_mask_cell, sampling=spacing)
                    signed = d_in - d_out
                    band = cell_mask & (np.abs(signed) <= pair.organelle_surface_band_um)
                    nonband = cell_mask & ~band
                    if np.any(band) and np.any(nonband):
                        band_mean = float(signal_full64[band].mean())
                        nonband_mean = float(signal_full64[nonband].mean())
                        eps = sig_mad if sig_mad > 0 else 0.0
                        num, den = band_mean + eps, nonband_mean + eps
                        if num > 0 and den > 0:
                            organelle_score = float(np.log2(num / den))

            row = out[cid]
            row[f"{prefix}__pearson_r"] = pearson_r
            row[f"{prefix}__manders_signal_in_reference"] = manders_signal_in_reference
            row[f"{prefix}__manders_reference_in_signal"] = manders_reference_in_signal
            row[f"{prefix}__organelle_surface_enrichment_score"] = organelle_score

    return out


@dataclass(frozen=True, slots=True)
class LocalizationDecisionInputs:
    """One cell's inputs to ``classify_localization``, for one signal channel.

    This is this module's own internal parameter bundle, assembled by the
    caller from ``compute_shell_enrichment_scores``,
    ``compute_colocalization`` and ``metrics.nuclei.compute_nuclei`` -- it is
    not a canonical ``contracts.py`` type.
    """

    total_corrected_signal: float | None
    extracellular_enrichment_score: float | None
    intracellular_enrichment_score: float | None
    has_nucleus: bool
    nuclear_enrichment: float | None
    organelle_surface_enrichment_score: float | None
    organelle_manders_signal_in_reference: float | None
    qc_passed: bool = True


def classify_localization(
    inputs: LocalizationDecisionInputs, decision: LocalizationDecisionConfig
) -> str:
    """One of ``extracellular_enriched``, ``intracellular_diffuse``,
    ``nuclear_enriched``, ``organelle_surface_associated``, ``mixed``,
    ``indeterminate``.

    Returns ``"indeterminate"`` unconditionally when ``decision.enabled`` is
    False -- see the module docstring for why -- and also on failed QC or
    insufficient corrected signal (``total_corrected_signal`` missing or
    below ``decision.minimum_corrected_signal``). The nuclear class is only
    reachable when ``inputs.has_nucleus`` is True (a nucleus mask actually
    exists for this cell); the organelle class requires BOTH its
    surface-enrichment score and its Manders score to clear their
    thresholds. Among classes that pass their threshold, the one with the
    largest margin above threshold wins if it leads the runner-up by at
    least ``decision.winning_margin``; otherwise (including a genuine tie)
    the call is ``"mixed"``. Exactly one class passing wins outright, with no
    runner-up to out-margin.
    """
    if not decision.enabled:
        return _INDETERMINATE
    if not inputs.qc_passed:
        return _INDETERMINATE
    if (
        inputs.total_corrected_signal is None
        or inputs.total_corrected_signal < decision.minimum_corrected_signal
    ):
        return _INDETERMINATE

    margins: dict[str, float] = {}

    if (
        inputs.extracellular_enrichment_score is not None
        and inputs.extracellular_enrichment_score >= decision.extracellular_log2_ratio_min
    ):
        margins["extracellular_enriched"] = (
            inputs.extracellular_enrichment_score - decision.extracellular_log2_ratio_min
        )

    if (
        inputs.intracellular_enrichment_score is not None
        and inputs.intracellular_enrichment_score >= decision.intracellular_log2_ratio_min
    ):
        margins["intracellular_diffuse"] = (
            inputs.intracellular_enrichment_score - decision.intracellular_log2_ratio_min
        )

    if (
        inputs.has_nucleus
        and inputs.nuclear_enrichment is not None
        and inputs.nuclear_enrichment >= decision.nuclear_log2_ratio_min
    ):
        margins["nuclear_enriched"] = (
            inputs.nuclear_enrichment - decision.nuclear_log2_ratio_min
        )

    if (
        inputs.organelle_surface_enrichment_score is not None
        and inputs.organelle_surface_enrichment_score >= decision.organelle_log2_ratio_min
        and inputs.organelle_manders_signal_in_reference is not None
        and inputs.organelle_manders_signal_in_reference >= decision.organelle_manders_min
    ):
        margins["organelle_surface_associated"] = (
            inputs.organelle_surface_enrichment_score - decision.organelle_log2_ratio_min
        )

    if not margins:
        return _INDETERMINATE
    if len(margins) == 1:
        return next(iter(margins))

    ranked = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)
    (top_class, top_margin), (_, second_margin) = ranked[0], ranked[1]
    if top_margin - second_margin >= decision.winning_margin:
        return top_class
    return _MIXED
