"""Canonical data structures shared by every stage of the pipeline.

Conventions frozen here (nothing downstream may deviate):

* Images are always ``CZYX``; instance label volumes are always ``ZYX``.
* Every physical tuple is ordered ``(z, y, x)`` to match array indexing.
* Distances are micrometres, areas um^2, volumes um^3. No pixel units escape
  into a physical field.
* Voxel centres sit at ``origin + (index + 0.5) * spacing``.
* Background label is 0; instances are positive ``uint32``. Label IDs need not be
  consecutive and are never silently renumbered on ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .errors import ContractViolation

Scalar: TypeAlias = str | int | float | bool | None
AxesCZYX: TypeAlias = Literal["CZYX"]
AxesZYX: TypeAlias = Literal["ZYX"]

SCHEMA_VERSION_CONFIG = "s2-pipeline-config/v1"
SCHEMA_VERSION_IMAGE = "s2-image-volume/v1"
SCHEMA_VERSION_LABELS = "s2-label-volume/v1"
SCHEMA_VERSION_RECORDS = "s2-records/v1"


class ChannelRole(StrEnum):
    """What a channel stains.

    Roles are configuration, never inferred from channel index or colour. No
    measurement function branches on a role; roles only decide which comparisons
    the orchestration layer asks for.
    """

    NUCLEUS = "nucleus"
    MEMBRANE = "membrane"
    CYTOPLASM = "cytoplasm"
    SIGNAL = "signal"
    ORGANELLE_MARKER = "organelle_marker"
    AUXILIARY = "auxiliary"
    IGNORE = "ignore"


class ContactEstimator(StrEnum):
    """How cell-cell interface area is estimated from voxelised labels.

    A first-class configuration choice, because the two estimators fail
    differently. All numbers below are measured against an analytic ground
    truth: two spheres (r=5um, centres 8um apart) split by their perpendicular
    bisector, so the true interface is a flat disc of pi*(r^2-(d/2)^2).

    ISOTROPIC sampling (0.1, 0.1, 0.1) um -- the ideal case:

        interface orientation      FACE_COUNT      MARCHING_CUBES
        axis-aligned                     0.6%                5.6%
        tilted 45 deg in XY             43.6%               13.4%
        tilted in all three axes        73.9%               10.2%
        ---------------------------------------------------------
        orientation spread              73.3pp               7.9pp

    FACE_COUNT is exact only for axis-aligned interfaces and inflates by up to
    sqrt(3) as the interface tilts. That is intrinsic digital-geometry staircase
    bias, and isotropic resampling does NOT remove it. MARCHING_CUBES is
    therefore the default.

    ANISOTROPIC sampling (0.5, 0.1, 0.1) um -- the REAL acquisition condition,
    MARCHING_CUBES, showing what ``resample_isotropic_before_contact`` buys:

        interface orientation      no resample     resampled (default)
        axis-aligned                    40.9%                    6.3%
        tilted 45 deg in XY             46.5%                   15.6%
        tilted in all three axes        71.3%                   45.9%
        --------------------------------------------------------------
        orientation spread              30.4pp                  39.6pp

    READ THAT SECOND TABLE BEFORE QUOTING A CONTACT AREA. Resampling roughly
    halves the error but cannot recover what the microscope never sampled: at 5x
    axial anisotropy an interface tilted toward the optical axis is still
    measured ~46% high. Because cells in an aggregate adhere at arbitrary
    orientations, this bias varies per contact with that contact's orientation,
    so it is confounded with packing geometry and does NOT cancel when comparing
    experimental conditions.

    Consequence for interpretation: absolute contact area at 5x anisotropy is a
    relative, orientation-sensitive quantity, not a trustworthy physical
    measurement. Coordination number and packing fraction are topological and
    volumetric respectively, and are far less orientation-sensitive.

    FACE_COUNT is retained for exact digital unit tests and sensitivity
    analysis, never as the headline adhesion metric.
    """

    MARCHING_CUBES = "marching_cubes"
    FACE_COUNT = "face_count"


@dataclass(frozen=True, slots=True)
class VoxelGeometry:
    """Physical sampling of a volume."""

    spacing_um_zyx: tuple[float, float, float]
    origin_um_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.spacing_um_zyx) != 3:
            raise ContractViolation("spacing_um_zyx must have exactly 3 elements")
        if any(s <= 0 for s in self.spacing_um_zyx):
            raise ContractViolation(
                f"spacing must be strictly positive, got {self.spacing_um_zyx}"
            )

    @property
    def voxel_volume_um3(self) -> float:
        dz, dy, dx = self.spacing_um_zyx
        return dz * dy * dx

    @property
    def face_area_um2_zyx(self) -> tuple[float, float, float]:
        """Area of the voxel face perpendicular to each axis, in (z, y, x) order."""
        dz, dy, dx = self.spacing_um_zyx
        return (dy * dx, dz * dx, dz * dy)

    @property
    def anisotropy_z_to_xy(self) -> float:
        """Z spacing divided by the finer in-plane spacing. 1.0 when isotropic."""
        dz, dy, dx = self.spacing_um_zyx
        return dz / min(dy, dx)

    @property
    def is_isotropic(self) -> bool:
        dz, dy, dx = self.spacing_um_zyx
        return max(dz, dy, dx) / min(dz, dy, dx) < 1.01

    def index_to_um(self, index_zyx: NDArray[np.floating]) -> NDArray[np.floating]:
        """Voxel indices to physical coordinates at voxel centres."""
        return (
            np.asarray(index_zyx, dtype=np.float64) + 0.5
        ) * np.asarray(self.spacing_um_zyx) + np.asarray(self.origin_um_zyx)


@dataclass(frozen=True, slots=True)
class ChannelBinding:
    """Binds a logical channel name to a physical index in the CZYX array.

    The optional labels carry experimental meaning that the pixels cannot:

    * ``color`` -- the acquisition colour (e.g. "green", "red"). Descriptive
      only; never used to infer biology.
    * ``population`` -- which cell GROUP this channel marks. Several channels may
      share one population (e.g. two co-transfected proteins both marking group
      "A"); two channels with different populations mark different cell groups,
      which is what an adhesion mixing assay reads out. ``None`` means the
      channel does not identify a population.
    * ``protein`` -- the tagged construct (e.g. "Cirl-GFP"). Free text, for
      labelling outputs and pairing co-localization channels.

    These are configuration, exactly like ``roles``: nothing in the measurement
    layer infers them from index, colour, or brightness. In particular a channel
    being brighter than another says nothing about which population a cell
    belongs to -- FITC routinely outshines mCherry several-fold.
    """

    channel_id: str
    source_index: int
    roles: frozenset[ChannelRole]
    source_name: str | None = None
    color: str | None = None
    population: str | None = None
    protein: str | None = None

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ContractViolation("channel_id must be non-empty")
        if self.source_index < 0:
            raise ContractViolation(
                f"source_index must be >= 0, got {self.source_index}"
            )

    def has_role(self, role: ChannelRole) -> bool:
        return role in self.roles


@dataclass(frozen=True, slots=True)
class FieldIdentity:
    """Stable identity of one field of view, carried across process boundaries."""

    dataset_id: str
    field_id: str
    source_uri: str
    source_field_index: int
    image_content_sha256: str


@dataclass(frozen=True, slots=True)
class ImageVolume:
    """A single field of view: CZYX intensities plus everything needed to measure."""

    data: NDArray[np.generic]
    geometry: VoxelGeometry
    channels: tuple[ChannelBinding, ...]
    identity: FieldIdentity
    axes: AxesCZYX = "CZYX"

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ContractViolation(
                f"ImageVolume.data must be 4-D CZYX, got shape {self.data.shape}"
            )
        n_c = self.data.shape[0]
        ids = [c.channel_id for c in self.channels]
        if len(set(ids)) != len(ids):
            raise ContractViolation(f"duplicate channel_id in {ids}")
        idxs = [c.source_index for c in self.channels]
        if len(set(idxs)) != len(idxs):
            raise ContractViolation(f"duplicate source_index in {idxs}")
        for c in self.channels:
            if c.source_index >= n_c:
                raise ContractViolation(
                    f"channel {c.channel_id!r} points at index {c.source_index} "
                    f"but the array has only {n_c} channels"
                )

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.data.shape[1:]  # type: ignore[return-value]

    def channel(self, channel_id: str) -> NDArray[np.generic]:
        """ZYX array for a logical channel. Raises if the id is not bound."""
        for c in self.channels:
            if c.channel_id == channel_id:
                return self.data[c.source_index]
        known = [c.channel_id for c in self.channels]
        raise ContractViolation(f"no channel {channel_id!r}; bound channels: {known}")

    def channels_with_role(self, role: ChannelRole) -> tuple[ChannelBinding, ...]:
        return tuple(c for c in self.channels if c.has_role(role))


@dataclass(frozen=True, slots=True)
class SegmentationProvenance:
    """Everything needed to reproduce, or distrust, a label volume."""

    run_id: str
    backend_id: str
    strategy: str
    config_sha256: str
    input_image_sha256: str
    device: str
    host_platform: str
    package_name: str | None = None
    package_version: str | None = None
    model_name: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LabelVolume:
    """Instance labels for one field. The unit of exchange between machines."""

    cells: NDArray[np.uint32]
    geometry: VoxelGeometry
    identity: FieldIdentity
    provenance: SegmentationProvenance
    nuclei: NDArray[np.uint32] | None = None
    axes: AxesZYX = "ZYX"

    def __post_init__(self) -> None:
        if self.cells.ndim != 3:
            raise ContractViolation(
                f"LabelVolume.cells must be 3-D ZYX, got shape {self.cells.shape}"
            )
        if self.cells.dtype != np.uint32:
            raise ContractViolation(
                f"labels must be uint32, got {self.cells.dtype}. Downcasting would "
                "silently merge instances."
            )
        if self.nuclei is not None:
            if self.nuclei.shape != self.cells.shape:
                raise ContractViolation(
                    f"nuclei shape {self.nuclei.shape} != cells {self.cells.shape}"
                )
            if self.nuclei.dtype != np.uint32:
                raise ContractViolation(f"nuclei must be uint32, got {self.nuclei.dtype}")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.cells.shape  # type: ignore[return-value]

    def cell_ids(self) -> NDArray[np.uint32]:
        """Sorted positive label ids present. Never renumbered."""
        u = np.unique(self.cells)
        return u[u > 0]

    def validate_against(self, image: ImageVolume | None) -> None:
        """Guard before any intensity or localization measurement.

        Geometry-only measurement is allowed with ``image=None``; intensity work
        is not, and callers must record ``missing_image_artifact`` in that case.
        """
        if image is None:
            return
        if image.shape_zyx != self.shape_zyx:
            raise ContractViolation(
                f"label shape {self.shape_zyx} != image shape {image.shape_zyx}"
            )
        if not np.allclose(
            image.geometry.spacing_um_zyx, self.geometry.spacing_um_zyx, rtol=1e-6
        ):
            raise ContractViolation(
                f"spacing mismatch: labels {self.geometry.spacing_um_zyx} vs "
                f"image {image.geometry.spacing_um_zyx}"
            )
        if image.identity.image_content_sha256 != self.provenance.input_image_sha256:
            raise ContractViolation(
                "these labels were produced from a different image "
                f"(labels claim {self.provenance.input_image_sha256[:12]}..., "
                f"image is {image.identity.image_content_sha256[:12]}...)"
            )


# ─── Output records ───────────────────────────────────────────────────────────
#
# `values` carries the open-ended metric payload so new metrics become new CSV
# columns without changing these classes. Identity and QC fields are explicit
# because they must never be optional.


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    """One row per cell (3D) or per legacy 2D aggregate."""

    dataset_id: str
    field_id: str
    backend_id: str
    object_kind: Literal["cell_3d", "legacy_aggregate_2d"]
    object_id: int
    touches_xy_border: bool
    touches_z_border: bool
    valid_for_geometry: bool
    values: Mapping[str, Scalar] = field(default_factory=dict)
    aggregate_id: int | None = None
    segmentation_run_id: str | None = None
    schema_version: str = SCHEMA_VERSION_RECORDS

    @property
    def is_truncated(self) -> bool:
        """Touching any of the six faces means the object is cropped.

        Volume, surface area and every hull-derived quantity are biased low for
        such objects, so canonical geometry fields are nulled and only
        ``observed_*`` values are kept.
        """
        return self.touches_xy_border or self.touches_z_border


@dataclass(frozen=True, slots=True)
class ContactRecord:
    """One row per unordered touching cell pair. ``cell_id_a < cell_id_b``."""

    dataset_id: str
    field_id: str
    segmentation_run_id: str
    cell_id_a: int
    cell_id_b: int
    contact_area_um2: float
    estimator: ContactEstimator
    qualifies_as_contact: bool
    valid_for_contact_metrics: bool
    values: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cell_id_a >= self.cell_id_b:
            raise ContractViolation(
                f"contact pair must be ordered a<b, got ({self.cell_id_a}, "
                f"{self.cell_id_b})"
            )


@dataclass(frozen=True, slots=True)
class AggregateRecord:
    """One row per connected component of the qualifying contact graph."""

    dataset_id: str
    field_id: str
    segmentation_run_id: str
    aggregate_id: int
    member_cell_ids: tuple[int, ...]
    contains_truncated_cell: bool
    values: Mapping[str, Scalar] = field(default_factory=dict)

    @property
    def cell_count(self) -> int:
        return len(self.member_cell_ids)


@dataclass(frozen=True, slots=True)
class LocalizationProfileRecord:
    """One row per cell x channel x reference x signed-distance bin."""

    dataset_id: str
    field_id: str
    cell_id: int
    channel_id: str
    reference: Literal["cell_boundary", "nucleus_boundary", "organelle_boundary"]
    bin_start_um: float
    bin_end_um: float
    voxel_count: int
    sampled_volume_um3: float
    mean_intensity: float | None = None
    integrated_corrected_intensity: float | None = None


@dataclass(frozen=True, slots=True)
class MeasurementBundle:
    """Everything one field yields."""

    objects: tuple[ObjectRecord, ...] = ()
    contacts: tuple[ContactRecord, ...] = ()
    aggregates: tuple[AggregateRecord, ...] = ()
    localization_profiles: tuple[LocalizationProfileRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class VolumeSource(Protocol):
    """Anything that can hand out fields of view: an nd2 file, a zarr store, a fake."""

    def field_ids(self) -> Sequence[str]: ...

    def read_field(self, field_id: str) -> ImageVolume: ...
