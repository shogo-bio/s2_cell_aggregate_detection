"""Configuration schema and YAML loading.

Every biological assumption lives here as explicit configuration. Nothing in the
measurement layer may infer a channel's meaning from its index, its colour, or
its brightness -- the code has no way to know what a channel stains, so it must
be told.

Unknown keys are errors, not warnings: a typo in a channel role would otherwise
silently disable a whole class of measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Literal,
    Mapping,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from .contracts import (
    SCHEMA_VERSION_CONFIG,
    ChannelBinding,
    ChannelRole,
    ContactEstimator,
)
from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    clip: bool = True


@dataclass(frozen=True, slots=True)
class CellposeEvalConfig:
    """Arguments forwarded to ``CellposeModel.eval()``.

    Every field here must correspond to a parameter the installed cellpose
    actually accepts. There used to be a ``tile: bool`` field, which neither
    cellpose 3.1.1.3 nor 4.2.1.1 has -- forwarding it raised
    ``TypeError: eval() got an unexpected keyword argument 'tile'`` and broke
    every real 3D run through the default backend. Tiling is unconditional in
    both versions; ``bsize`` and ``tile_overlap`` are the real knobs.

    ``None`` means "do not pass this argument", leaving the library default,
    which keeps us from pinning a value cellpose may retune between releases.
    """

    flow_threshold: float | None = 0.4
    cellprob_threshold: float | None = 0.0
    bsize: int | None = None
    tile_overlap: float | None = None
    augment: bool = False
    batch_size: int = 8
    # 2.5D stitching. When set, cellpose segments each Z plane in 2D (in the fine
    # XY resolution) and stitches the planes into 3D instances by IoU >= this
    # value, instead of a native 3D flow (do_3D). For strongly anisotropic
    # stacks -- this data is 2 um in Z vs 0.63 um in XY -- 2.5D is both faster
    # and more reliable, because it never asks cellpose to resolve shapes along
    # the poorly sampled axis. ``None`` uses native 3D (do_3D=True).
    stitch_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class CellposeModelConfig:
    """Which cellpose to use.

    The local default is package_major=3 with cyto3 on CPU. That is an
    operational choice forced by measured runtime -- cellpose 4's cpsam needed
    638 s for a single 256x256 plane on this host versus 2.2 s for cyto3 -- not a
    claim that cyto3 segments better. See docs/cellpose_cpu_notes.md.
    """

    package_major: Literal[3, 4] = 3
    model_name: str = "cyto3"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    diameter_um: float | None = None
    pretrained_model_path: Path | None = None
    expected_model_sha256: str | None = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    eval: CellposeEvalConfig = field(default_factory=CellposeEvalConfig)


ChannelCombination: TypeAlias = Literal["stack", "max", "sum"]
CHANNEL_COMBINATIONS: tuple[str, ...] = get_args(ChannelCombination)


@dataclass(frozen=True, slots=True)
class DirectInstanceConfig:
    """Cellpose on the configured channels, no nuclear seeding.

    ``channel_combination`` decides what cellpose actually sees:

    * ``"stack"`` (default, historical behaviour): the selected channels are
      handed over as separate channels (cellpose-3: first = cytoplasm,
      second = nucleus, at most two).
    * ``"max"`` / ``"sum"``: each channel is percentile-normalised on its own
      and the normalised channels are then merged into ONE grayscale image
      (voxel-wise maximum, or sum re-normalised to the same range). Use this
      when the populations to segment are marked by *different* channels --
      e.g. Cirl-GFP cells vs Cirl-mCherry cells -- so that a cell bright in
      only one of them still has an outline for cellpose to find. Feeding the
      channels stacked would instead tell cellpose-3 the second channel is a
      nucleus, which it is not.
    """

    strategy: Literal["direct_cellpose"]
    input_channel_ids: tuple[str, ...]
    model: CellposeModelConfig = field(default_factory=CellposeModelConfig)
    channel_combination: ChannelCombination = "stack"
    min_cell_volume_um3: float = 50.0
    fill_internal_holes: bool = True
    split_disconnected_labels: bool = True


@dataclass(frozen=True, slots=True)
class NucleusSeededWatershedConfig:
    """Nucleus-seeded watershed.

    Preferred wherever a nuclear marker exists: nuclei are well separated and
    near-convex, so they segment far more reliably than touching cell bodies --
    and dense aggregates, the region of interest here, are exactly where direct
    cell instance segmentation fails.
    """

    strategy: Literal["nucleus_seeded_watershed"]
    nucleus_channel_id: str
    boundary_channel_ids: tuple[str, ...]
    seed_model: CellposeModelConfig = field(default_factory=CellposeModelConfig)
    extent_mode: Literal["cellpose_union", "adaptive_intensity"] = "adaptive_intensity"
    extent_model: CellposeModelConfig | None = None
    gaussian_sigma_um: float = 0.5
    watershed_compactness: float = 0.0
    min_nucleus_volume_um3: float = 20.0
    min_cell_volume_um3: float = 50.0
    max_nuclei_per_cell: int = 1


@dataclass(frozen=True, slots=True)
class StarDistModelConfig:
    """Which StarDist model to use, and how to feed it.

    StarDist differs from Cellpose in one way that matters here: anisotropy is a
    TRAINING parameter baked into the model (``Config3D(anisotropy=...)``), not
    something you pass at prediction time. A model therefore expects input
    sampled the way its training data was. ``resample_isotropic`` handles the
    common case of a model trained on isotropic data by resampling the volume
    before inference and mapping labels back to the acquired grid afterwards;
    set it False when using a model trained at this experiment's own anisotropy.

    ``pretrained_name`` selects a published model; ``custom_model_dir`` plus
    ``custom_model_name`` load a locally trained or fine-tuned one. Exactly one
    of the two must be given -- there is no silent fallback, because quietly
    running a nuclear model on membrane-labelled cells would produce confident,
    well-formed, wrong instances.
    """

    pretrained_name: str | None = None
    custom_model_dir: Path | None = None
    custom_model_name: str | None = None
    normalization: NormalizationConfig = field(
        default_factory=lambda: NormalizationConfig(lower_percentile=1.0,
                                                   upper_percentile=99.8)
    )
    prob_threshold: float | None = None
    nms_threshold: float | None = None
    n_tiles_zyx: tuple[int, int, int] | None = None
    resample_isotropic: bool = True


@dataclass(frozen=True, slots=True)
class DirectStarDistConfig:
    """3D instance segmentation with StarDist.

    Worth trying alongside Cellpose for this data: the signal is a membrane
    shell, so the bright ring IS the boundary StarDist predicts distances to,
    and the star-convex shape prior actively helps where the ring is incomplete
    -- it enforces a closed outline across gaps rather than letting a flow field
    leak through them. S2 cells are close to spherical, so the star-convexity
    assumption costs little.
    """

    strategy: Literal["direct_stardist"]
    input_channel_ids: tuple[str, ...]
    model: StarDistModelConfig = field(default_factory=StarDistModelConfig)
    min_cell_volume_um3: float = 50.0
    fill_internal_holes: bool = True
    split_disconnected_labels: bool = True


SegmentationConfig: TypeAlias = (
    DirectInstanceConfig | NucleusSeededWatershedConfig | DirectStarDistConfig
)


@dataclass(frozen=True, slots=True)
class OpticsConfig:
    """Measured point-spread function, used to state what is resolvable.

    These are not cosmetic. Two cell membranes pressed together in an aggregate
    sit closer than the axial PSF, so the fluorescence there cannot be assigned
    to either cell from the image alone -- the two-source model
    ``I = a_A*h(s+d/2) + a_B*h(s-d/2)`` is unidentifiable at small d, where only
    the SUM is estimable. The pipeline therefore flags such contacts rather than
    attempting to split them, and needs the PSF width to know when to do so.

    Defaults are typical confocal values. Measure your own with sub-resolution
    beads under the actual objective, wavelength, pinhole, immersion medium and
    depth -- published numbers are a starting point, not a substitute.
    """

    axial_fwhm_um: float = 0.7
    lateral_fwhm_um: float = 0.25

    def resolution_along_um(self, normal_zyx: tuple[float, float, float]) -> float:
        """Effective resolution in the direction of a unit normal.

        An interface facing the optical axis is resolved at the (poor) axial
        width; one lying in the imaging plane gets the (good) lateral width.
        """
        nz, ny, nx = normal_zyx
        return float(
            (
                self.lateral_fwhm_um**2 * (ny**2 + nx**2)
                + self.axial_fwhm_um**2 * nz**2
            )
            ** 0.5
        )


@dataclass(frozen=True, slots=True)
class SaturationConfig:
    """Detector clipping and dynamic-range adequacy checks.

    Real acquisitions here were found with one channel clipping at the detector
    maximum (FITC, ~13% of bright pixels at 4095) and another barely using the
    range (mCherry, ~22%). Both silently corrupt intensity-derived metrics --
    a clipped ring reads flatter than it is, a dim channel's background-relative
    thresholds are noise -- so each cell/channel carries a saturation fraction
    and a dynamic-range flag rather than pretending the intensities are clean.

    ``detector_max``: the clipping value. ``None`` auto-detects it from the data
    (a spike sitting exactly on a power-of-two-minus-one). Set it explicitly
    (e.g. 4095 for a 12-bit detector, 65535 for 16-bit) when you know it, since
    auto-detection fails on data that never actually reaches the ceiling.
    """

    detector_max: int | None = None
    # A cell/channel with more than this fraction of its voxels at the detector
    # max is flagged saturated; the policy for what that invalidates lives in the
    # metrics layer.
    saturated_fraction_threshold: float = 0.02
    # A channel whose 99.9th percentile sits below this fraction of the detector
    # range is flagged under-exposed -- too close to the noise floor to trust.
    min_dynamic_range_fraction: float = 0.10


@dataclass(frozen=True, slots=True)
class ContactConfig:
    """How cell-cell interface area is measured.

    ``estimator`` and ``resample_isotropic_before_contact`` default as they do
    because of measured orientation bias -- see ContactEstimator's docstring for
    the numbers. Changing them changes what the headline adhesion metric means.
    """

    estimator: ContactEstimator = ContactEstimator.MARCHING_CUBES
    resample_isotropic_before_contact: bool = True
    minimum_contact_area_um2: float = 1.0
    report_proximity_within_um: float | None = None


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    mode: Literal["fixed", "outside_cells_median_mad"] = "outside_cells_median_mad"
    fixed_value_by_channel: Mapping[str, float] = field(default_factory=dict)
    mad_multiplier: float = 3.0


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    mode: Literal["fixed", "background_mad"] = "background_mad"
    fixed_value: float | None = None
    mad_multiplier: float | None = 3.0


@dataclass(frozen=True, slots=True)
class ColocalizationPairConfig:
    """Two channels whose overlap within each cell is measured (Pearson, Manders).

    ``reference_role`` ``"protein"`` is the co-transfection case: two tagged
    proteins expressed in the SAME cell population, where the question is how
    much they overlap sub-cellularly. ``"nucleus"``/``"organelle_marker"`` ask
    whether a signal concentrates at a named compartment. The three differ only
    in what the reference channel is required to be, not in the maths.
    """

    signal_channel_id: str
    reference_channel_id: str
    reference_role: Literal["nucleus", "organelle_marker", "protein"]
    signal_threshold: ThresholdSpec = field(default_factory=ThresholdSpec)
    reference_threshold: ThresholdSpec = field(default_factory=ThresholdSpec)
    organelle_surface_band_um: float | None = None


@dataclass(frozen=True, slots=True)
class LocalizationDecisionConfig:
    """Categorical localization calls.

    Disabled by default and must stay disabled until thresholds are calibrated
    against real stacks with biological controls. Axial optical resolution is
    ~0.5-0.8 um while a membrane is ~5 nm thick, so no call here is a statement
    about which side of a membrane a molecule sits on -- these are
    resolution-limited enrichment calls. The continuous profiles underneath are
    always available and need no calibration.
    """

    enabled: bool = False
    minimum_corrected_signal: float = 0.0
    extracellular_log2_ratio_min: float = 1.0
    intracellular_log2_ratio_min: float = 1.0
    nuclear_log2_ratio_min: float = 1.0
    organelle_log2_ratio_min: float = 1.0
    organelle_manders_min: float = 0.5
    winning_margin: float = 0.5


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    signed_distance_bin_edges_um: tuple[float, ...] = (
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0,
    )
    inner_shell_width_um: float = 1.0
    outer_shell_width_um: float = 1.0
    colocalization_pairs: tuple[ColocalizationPairConfig, ...] = ()
    decision: LocalizationDecisionConfig = field(
        default_factory=LocalizationDecisionConfig
    )


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    saturation: SaturationConfig = field(default_factory=SaturationConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    compute_surface_meshes: bool = True
    compute_single_cell_hulls: bool = True


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    format: Literal["zarr", "ome-tiff"] = "zarr"
    image_chunks_czyx: tuple[int, int, int, int] = (1, 8, 256, 256)
    label_chunks_zyx: tuple[int, int, int] = (8, 256, 256)
    compression_level: int = 3
    verify_content_hashes: bool = True


@dataclass(frozen=True, slots=True)
class LegacyConfig:
    """Frozen parameters of the original 2D threshold pipeline, kept for comparison."""

    threshold_uint8: int = 50
    s2_diameter_um: float = 10.0
    minimum_cell_equivalents: float = 3.0
    closing_radius_um: float = 1.65
    minimum_active_channels: int = 2


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    channels: tuple[ChannelBinding, ...]
    analysis_backend: Literal["legacy_threshold_2d", "ml_instance_3d"] = "ml_instance_3d"
    segmentation: SegmentationConfig | None = None
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    legacy: LegacyConfig | None = None
    schema_version: str = SCHEMA_VERSION_CONFIG


# ─── Loading ──────────────────────────────────────────────────────────────────


def _build(cls: type, raw: Any, where: str) -> Any:
    """Recursively construct a frozen dataclass from plain YAML data."""
    if not is_dataclass(cls):
        return raw
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. Known keys: {sorted(known)}"
        )

    # `from __future__ import annotations` leaves every annotation a string, so
    # resolve them for real rather than pattern-matching the source text.
    hints = get_type_hints(cls, globalns=globals())

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        kwargs[name] = _coerce(hints[name], value, f"{where}.{name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _coerce(ftype: Any, value: Any, where: str) -> Any:
    """Turn YAML scalars/sequences into the declared field type."""
    origin = get_origin(ftype)

    # Optional[X] / X | None -> use X when a value is actually present
    if origin is UnionType or origin is Union:
        candidates = [a for a in get_args(ftype) if a is not type(None)]
        if value is None:
            return None
        if len(candidates) == 1:
            return _coerce(candidates[0], value, where)
        return value

    if origin is tuple and isinstance(value, list):
        args = [a for a in get_args(ftype) if a is not Ellipsis]
        inner = args[0] if args else Any
        return tuple(_coerce(inner, v, where) for v in value)

    if is_dataclass(ftype) and isinstance(value, Mapping):
        return _build(ftype, value, where)

    if isinstance(ftype, type) and issubclass(ftype, StrEnum):
        try:
            return ftype(value)
        except ValueError as exc:
            opts = [e.value for e in ftype]
            raise ConfigError(f"{where}: {value!r} is not one of {opts}") from exc

    if ftype is Path and isinstance(value, str):
        return Path(value)

    return value


def _parse_channels(raw: Any) -> tuple[ChannelBinding, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("channels: must be a non-empty list")
    out: list[ChannelBinding] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"channels[{i}]: expected a mapping")
        allowed = {
            "channel_id", "source_index", "roles", "source_name",
            "color", "population", "protein",
        }
        unknown = set(entry) - allowed
        if unknown:
            raise ConfigError(f"channels[{i}]: unknown key(s) {sorted(unknown)}")
        roles_raw = entry.get("roles", [])
        if isinstance(roles_raw, str):
            roles_raw = [roles_raw]
        try:
            roles = frozenset(ChannelRole(r) for r in roles_raw)
        except ValueError as exc:
            raise ConfigError(
                f"channels[{i}].roles: {exc}. Valid roles: "
                f"{[r.value for r in ChannelRole]}"
            ) from exc
        try:
            out.append(
                ChannelBinding(
                    channel_id=entry["channel_id"],
                    source_index=int(entry["source_index"]),
                    roles=roles,
                    source_name=entry.get("source_name"),
                    color=entry.get("color"),
                    population=entry.get("population"),
                    protein=entry.get("protein"),
                )
            )
        except KeyError as exc:
            raise ConfigError(f"channels[{i}]: missing required key {exc}") from exc
    return tuple(out)


def _parse_segmentation(raw: Any) -> SegmentationConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or "strategy" not in raw:
        raise ConfigError("segmentation: must be a mapping with a 'strategy' key")
    strategy = raw["strategy"]
    if strategy == "direct_cellpose":
        return _build(DirectInstanceConfig, raw, "segmentation")
    if strategy == "nucleus_seeded_watershed":
        return _build(NucleusSeededWatershedConfig, raw, "segmentation")
    if strategy == "direct_stardist":
        return _build(DirectStarDistConfig, raw, "segmentation")
    raise ConfigError(
        f"segmentation.strategy: {strategy!r} is not one of "
        "['direct_cellpose', 'nucleus_seeded_watershed', 'direct_stardist']"
    )


def load_config(path: Path) -> PipelineConfig:
    """Read and fully validate a YAML config."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")

    known = {f.name for f in fields(PipelineConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {sorted(unknown)}")

    version = raw.get("schema_version", SCHEMA_VERSION_CONFIG)
    if version != SCHEMA_VERSION_CONFIG:
        raise ConfigError(
            f"schema_version {version!r} is not supported "
            f"(this build reads {SCHEMA_VERSION_CONFIG!r})"
        )

    cfg = PipelineConfig(
        channels=_parse_channels(raw.get("channels")),
        analysis_backend=raw.get("analysis_backend", "ml_instance_3d"),
        segmentation=_parse_segmentation(raw.get("segmentation")),
        measurement=_build(
            MeasurementConfig, raw.get("measurement", {}), "measurement"
        ),
        artifacts=_build(ArtifactConfig, raw.get("artifacts", {}), "artifacts"),
        legacy=_build(LegacyConfig, raw["legacy"], "legacy")
        if raw.get("legacy") is not None
        else None,
    )
    validate_config(cfg)
    return cfg


def validate_config(config: PipelineConfig) -> None:
    """Cross-field checks that a per-field schema cannot express."""
    ids = [c.channel_id for c in config.channels]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"duplicate channel_id: {ids}")
    idxs = [c.source_index for c in config.channels]
    if len(set(idxs)) != len(idxs):
        raise ConfigError(f"duplicate source_index: {idxs}")
    known = set(ids)

    def require(cid: str, why: str) -> None:
        if cid not in known:
            raise ConfigError(f"{why} names channel {cid!r}, which is not configured "
                              f"(configured: {sorted(known)})")

    seg = config.segmentation
    if config.analysis_backend == "ml_instance_3d" and seg is None:
        raise ConfigError("analysis_backend 'ml_instance_3d' requires a segmentation block")
    if config.analysis_backend == "legacy_threshold_2d" and config.legacy is None:
        raise ConfigError("analysis_backend 'legacy_threshold_2d' requires a legacy block")

    if isinstance(seg, DirectInstanceConfig):
        if not seg.input_channel_ids:
            raise ConfigError("direct_cellpose: input_channel_ids must be non-empty")
        for cid in seg.input_channel_ids:
            require(cid, "segmentation.input_channel_ids")
        if seg.channel_combination not in CHANNEL_COMBINATIONS:
            raise ConfigError(
                f"segmentation.channel_combination: {seg.channel_combination!r} "
                f"is not one of {list(CHANNEL_COMBINATIONS)}"
            )
        if seg.channel_combination == "stack":
            if seg.model.package_major == 3 and len(seg.input_channel_ids) > 2:
                raise ConfigError(
                    "cellpose 3 accepts at most two input channels, got "
                    f"{len(seg.input_channel_ids)}"
                )
        elif len(seg.input_channel_ids) < 2:
            raise ConfigError(
                f"segmentation.channel_combination {seg.channel_combination!r} "
                "merges several channels into one image, so input_channel_ids "
                f"needs at least two entries, got {list(seg.input_channel_ids)}"
            )
    elif isinstance(seg, DirectStarDistConfig):
        if not seg.input_channel_ids:
            raise ConfigError("direct_stardist: input_channel_ids must be non-empty")
        for cid in seg.input_channel_ids:
            require(cid, "segmentation.input_channel_ids")
        if len(seg.input_channel_ids) != 1:
            raise ConfigError(
                "direct_stardist takes exactly one input channel, got "
                f"{len(seg.input_channel_ids)}: StarDist 3D predicts from a "
                "single intensity volume."
            )
        model = seg.model
        has_pretrained = model.pretrained_name is not None
        has_custom = model.custom_model_dir is not None
        if has_pretrained == has_custom:
            raise ConfigError(
                "segmentation.model: give exactly one of pretrained_name or "
                "custom_model_dir. Published StarDist models are trained on "
                "nuclei; running one on membrane-labelled cells without saying "
                "so explicitly would produce confident but meaningless instances."
            )
        if has_custom and not model.custom_model_name:
            raise ConfigError(
                "segmentation.model.custom_model_name is required alongside "
                "custom_model_dir (StarDist loads a model by name from a "
                "base directory)."
            )

    elif isinstance(seg, NucleusSeededWatershedConfig):
        require(seg.nucleus_channel_id, "segmentation.nucleus_channel_id")
        nuc = next(c for c in config.channels if c.channel_id == seg.nucleus_channel_id)
        if not nuc.has_role(ChannelRole.NUCLEUS):
            raise ConfigError(
                f"nucleus_seeded_watershed names {seg.nucleus_channel_id!r} as the "
                "nucleus channel, but that channel does not carry the 'nucleus' role. "
                "Segmenting a non-nuclear channel as seeds produces meaningless "
                "instances -- fix the roles rather than this reference."
            )
        for cid in seg.boundary_channel_ids:
            require(cid, "segmentation.boundary_channel_ids")
        if seg.extent_mode == "cellpose_union" and seg.extent_model is None:
            raise ConfigError("extent_mode 'cellpose_union' requires extent_model")

    loc = config.measurement.localization
    edges = loc.signed_distance_bin_edges_um
    if len(edges) < 2:
        raise ConfigError("signed_distance_bin_edges_um needs at least two edges")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ConfigError(f"signed_distance_bin_edges_um must strictly increase: {edges}")

    by_id = {c.channel_id: c for c in config.channels}
    for i, pair in enumerate(loc.colocalization_pairs):
        require(pair.signal_channel_id, f"colocalization_pairs[{i}].signal_channel_id")
        require(pair.reference_channel_id, f"colocalization_pairs[{i}].reference_channel_id")
        sig = by_id[pair.signal_channel_id]
        ref = by_id[pair.reference_channel_id]

        if pair.reference_role == "protein":
            # Co-transfection overlap: two tagged proteins in the SAME cell
            # population. Both should be signal channels, and pairing across two
            # populations would be meaningless -- those are different cells.
            if not ref.has_role(ChannelRole.SIGNAL):
                raise ConfigError(
                    f"colocalization_pairs[{i}] declares reference_role 'protein' "
                    f"but channel {ref.channel_id!r} does not carry the 'signal' "
                    "role. A protein-protein colocalization reference must be a "
                    "signal channel."
                )
            if (
                sig.population is not None
                and ref.population is not None
                and sig.population != ref.population
            ):
                raise ConfigError(
                    f"colocalization_pairs[{i}] pairs channels from different "
                    f"populations ({sig.population!r} vs {ref.population!r}). "
                    "Colocalization measures overlap WITHIN a cell; two "
                    "populations are different cells. Use the adhesion mixing "
                    "readout for between-population questions instead."
                )
        else:
            wanted = ChannelRole(pair.reference_role)
            if not ref.has_role(wanted):
                raise ConfigError(
                    f"colocalization_pairs[{i}] declares reference_role "
                    f"{wanted.value!r} but channel {ref.channel_id!r} has roles "
                    f"{sorted(r.value for r in ref.roles)}"
                )

    c = config.measurement.contact
    if c.minimum_contact_area_um2 < 0:
        raise ConfigError("minimum_contact_area_um2 must be >= 0")
    if c.estimator is ContactEstimator.FACE_COUNT and c.resample_isotropic_before_contact:
        raise ConfigError(
            "face_count with resample_isotropic_before_contact=true is not a "
            "meaningful combination: resampling does not reduce face-count's "
            "orientation bias (measured 73.3pp spread isotropic vs 70.7pp "
            "anisotropic). Use marching_cubes, or set resampling off and treat "
            "face_count output as a digital sensitivity check only."
        )
