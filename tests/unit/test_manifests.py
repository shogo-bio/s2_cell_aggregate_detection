"""Tests for s2_adhesion.io.manifests: schema validation, canonical JSON,
and the artifact-binding guard.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from s2_adhesion.contracts import (
    ChannelBinding,
    ChannelRole,
    FieldIdentity,
    ImageVolume,
    VoxelGeometry,
)
from s2_adhesion.errors import ArtifactBindingError, ArtifactError
from s2_adhesion.io.manifests import (
    build_image_manifest,
    build_label_manifest,
    canonical_json,
    check_label_image_binding,
    image_channels_from_manifest,
    image_identity_from_manifest,
    label_identity_from_manifest,
    provenance_from_manifest,
    sha256_of_array,
    validate_image_manifest,
    validate_label_manifest,
)

from tests.conftest import ANISOTROPIC, make_image_volume, make_label_volume


# ─── canonical JSON / hashing ───────────────────────────────────────────────


def test_canonical_json_is_sorted_and_deterministic():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    text1 = canonical_json(a)
    text2 = canonical_json(copy.deepcopy(a))
    assert text1 == text2
    # key order in the source dict must not matter
    reordered = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(reordered) == text1
    # sorted keys really are sorted, at every nesting level
    parsed_text = text1.splitlines()
    assert any('"a"' in line for line in parsed_text)
    idx_a = text1.index('"a"')
    idx_b = text1.index('"b"')
    idx_c = text1.index('"c"')
    assert idx_a < idx_b < idx_c


def test_sha256_of_array_ignores_memory_layout_not_content():
    arr = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    transposed_back = np.ascontiguousarray(arr.T).T  # non-contiguous, same content
    assert sha256_of_array(arr) == sha256_of_array(transposed_back)
    other = arr.copy()
    other[0, 0, 0] = 12345
    assert sha256_of_array(arr) != sha256_of_array(other)
    # sanity: matches a plain hashlib computation on contiguous bytes
    expected = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    assert sha256_of_array(arr) == expected


# ─── image manifest build/validate round trip ──────────────────────────────


def test_build_and_validate_image_manifest_round_trip():
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    vol = make_image_volume(data, spacing=ANISOTROPIC)
    manifest = build_image_manifest(
        vol, artifact_id="img-abc", array_sha256="f" * 64, created_utc="2026-01-01T00:00:00+00:00"
    )
    # round trips through JSON exactly (manifest must be plain-JSON-safe)
    reparsed = json.loads(canonical_json(manifest))
    validated = validate_image_manifest(reparsed)
    assert validated["dataset_id"] == vol.identity.dataset_id
    assert validated["field_id"] == vol.identity.field_id
    assert tuple(validated["spacing_um_zyx"]) == ANISOTROPIC
    assert validated["shape"] == list(data.shape)
    assert validated["dtype"] == "uint16"

    identity = image_identity_from_manifest(validated)
    assert identity == vol.identity
    channels = image_channels_from_manifest(validated)
    assert channels == vol.channels


def test_validate_image_manifest_rejects_missing_key():
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    vol = make_image_volume(data, spacing=ANISOTROPIC)
    manifest = build_image_manifest(
        vol, artifact_id="img-abc", array_sha256="f" * 64, created_utc="now"
    )
    incomplete = dict(manifest)
    del incomplete["spacing_um_zyx"]
    with pytest.raises(ArtifactError):
        validate_image_manifest(incomplete)


def test_validate_image_manifest_rejects_wrong_schema_version():
    data = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    vol = make_image_volume(data, spacing=ANISOTROPIC, channel_ids=("membrane",),
                             roles=(ChannelRole.MEMBRANE,))
    manifest = build_image_manifest(
        vol, artifact_id="img-abc", array_sha256="f" * 64, created_utc="now"
    )
    bad = dict(manifest)
    bad["schema_version"] = "s2-image-volume/v999"
    with pytest.raises(ArtifactError):
        validate_image_manifest(bad)


# ─── label manifest build/validate round trip ──────────────────────────────


def test_build_and_validate_label_manifest_round_trip():
    cells = np.zeros((3, 4, 5), dtype=np.uint32)
    cells[0, 0, 0] = 1
    cells[1, 1, 1] = 5
    cells[2, 2, 2] = 900
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    manifest = build_label_manifest(
        lab,
        artifact_id="lab-abc",
        cells_sha256=sha256_of_array(cells),
        nuclei_sha256=None,
        created_utc="2026-01-01T00:00:00+00:00",
        input_image_artifact_id="img-xyz",
    )
    reparsed = json.loads(canonical_json(manifest))
    validated = validate_label_manifest(reparsed)
    assert validated["dtype"] == "uint32"
    assert validated["shape"] == [3, 4, 5]
    cells_entry = next(e for e in validated["labels"] if e["name"] == "cells")
    assert cells_entry["present"] is True
    nuclei_entry = next(e for e in validated["labels"] if e["name"] == "nuclei")
    assert nuclei_entry["present"] is False

    identity = label_identity_from_manifest(validated)
    assert identity == lab.identity
    provenance = provenance_from_manifest(validated)
    assert provenance == lab.provenance


def test_validate_label_manifest_rejects_non_uint32_dtype():
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    manifest = build_label_manifest(
        lab,
        artifact_id="lab-abc",
        cells_sha256=sha256_of_array(cells),
        nuclei_sha256=None,
        created_utc="now",
        input_image_artifact_id=None,
    )
    tampered = dict(manifest)
    tampered["dtype"] = "uint16"
    with pytest.raises(ArtifactError):
        validate_label_manifest(tampered)


def test_validate_label_manifest_requires_cells_entry_present():
    cells = np.zeros((2, 2, 2), dtype=np.uint32)
    lab = make_label_volume(cells, spacing=ANISOTROPIC)
    manifest = build_label_manifest(
        lab,
        artifact_id="lab-abc",
        cells_sha256=sha256_of_array(cells),
        nuclei_sha256=None,
        created_utc="now",
        input_image_artifact_id=None,
    )
    tampered = copy.deepcopy(manifest)
    tampered["labels"] = [e for e in tampered["labels"] if e["name"] != "cells"]
    with pytest.raises(ArtifactError):
        validate_label_manifest(tampered)


# ─── binding guard ──────────────────────────────────────────────────────────


def _paired_image_and_label(shape_zyx=(3, 4, 5), spacing=ANISOTROPIC):
    cells = np.zeros(shape_zyx, dtype=np.uint32)
    cells[0, 0, 0] = 1
    lab = make_label_volume(cells, spacing=spacing)
    # build an image whose content hash equals what the label's provenance claims
    image_bytes = b"the-source-image"
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    identity = FieldIdentity(
        dataset_id="synthetic",
        field_id="field00",
        source_uri="memory://synthetic",
        source_field_index=0,
        image_content_sha256=image_sha,
    )
    data = np.zeros((1, *shape_zyx), dtype=np.uint16)
    image = ImageVolume(
        data=data,
        geometry=VoxelGeometry(spacing_um_zyx=spacing),
        channels=(
            ChannelBinding(
                channel_id="membrane", source_index=0, roles=frozenset({ChannelRole.MEMBRANE})
            ),
        ),
        identity=identity,
    )
    # rebuild the label so its provenance.input_image_sha256 matches this image
    from dataclasses import replace

    lab = replace(lab, provenance=replace(lab.provenance, input_image_sha256=image_sha))
    return lab, image


def test_check_label_image_binding_passes_for_matching_pair():
    lab, image = _paired_image_and_label()
    check_label_image_binding(lab, image)  # must not raise


def test_check_label_image_binding_rejects_shape_mismatch():
    lab, image = _paired_image_and_label(shape_zyx=(3, 4, 5))
    _, other_image = _paired_image_and_label(shape_zyx=(3, 4, 6))
    # reuse lab's provenance hash on a differently-shaped image
    from dataclasses import replace

    mismatched_image = replace(
        other_image,
        identity=replace(other_image.identity, image_content_sha256=lab.provenance.input_image_sha256),
    )
    with pytest.raises(ArtifactBindingError, match="shape"):
        check_label_image_binding(lab, mismatched_image)


def test_check_label_image_binding_rejects_spacing_mismatch():
    lab, image = _paired_image_and_label(spacing=ANISOTROPIC)
    from dataclasses import replace

    mismatched_image = replace(
        image, geometry=VoxelGeometry(spacing_um_zyx=(0.2, 0.2, 0.2))
    )
    with pytest.raises(ArtifactBindingError, match="spacing"):
        check_label_image_binding(lab, mismatched_image)


def test_check_label_image_binding_rejects_image_hash_mismatch():
    lab, image = _paired_image_and_label()
    from dataclasses import replace

    mismatched_image = replace(
        image, identity=replace(image.identity, image_content_sha256="0" * 64)
    )
    with pytest.raises(ArtifactBindingError, match="different image"):
        check_label_image_binding(lab, mismatched_image)


def test_manifests_module_does_not_import_torch_or_cellpose():
    """Must run in a fresh interpreter.

    An in-process sys.modules check passes or fails depending on what earlier
    tests in the session happened to import, so it tests collection order rather
    than this module. A subprocess is the only honest way to ask the question.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import s2_adhesion.io.manifests; "
            "leaked = [m for m in sys.modules if m.split('.')[0] in ('torch', 'cellpose')]; "
            "print(leaked); sys.exit(1 if leaked else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing s2_adhesion.io.manifests pulled in ML packages: {result.stdout.strip()}"
    )
