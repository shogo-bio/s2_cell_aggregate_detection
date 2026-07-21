# Artifact contract

This document describes the on-disk boundary between segmentation and
measurement: `s2_adhesion.io.manifests`, `s2_adhesion.io.zarr_store`, and
`s2_adhesion.io.tiff_store`.

## Why this boundary exists

Segmentation is far too slow to run on a laptop. On the reference
development host, cellpose 4 (`cpsam`) needed 638 s for a single 256x256
plane; cellpose 3 (`cyto3`) is roughly 290x faster but still slow enough that
whole-volume segmentation belongs on a GPU box, not the machine doing
measurement and plotting. So segmentation and measurement routinely run on
**different machines**, and a label volume has to survive that trip without
silently losing or corrupting anything: spacing, origin, non-consecutive
label ids, which image it was segmented from, and how it was produced.

These artifacts are that boundary. `s2_adhesion.contracts.LabelVolume` and
`ImageVolume` are the in-memory shapes; this module is how they get to disk
and back intact, or get rejected outright when they can't be trusted.

## The three-machine workflow

```
acquisition machine              GPU machine                  laptop / CI
------------------              ------------                  -----------
 .nd2 file
   |  extract()
   v
image.ome.zarr  ---------------->  read_image_volume()
                                     |  segment (cellpose)
                                     v
                                   labels.ome.zarr  --------->  read_label_volume(path, image=...)
                                                                  |  check_label_image_binding
                                                                  v
                                                                measurement (no torch/cellpose needed)
```

Each arrow is a plain file copy (rsync, a shared drive, cloud storage --
nothing artifact-specific). The receiving side never has to trust the
sender: every reader independently verifies completeness (`_SUCCESS`),
content (`array_sha256`), and, when an image is supplied, that the labels
actually describe that image (`check_label_image_binding`).

`s2_adhesion.io.zarr_store` and `s2_adhesion.io.tiff_store` both import
cleanly with no `torch`/`cellpose` installed -- verified by a subprocess test
in each `tests/unit/test_*_store.py` -- because the measurement side of this
workflow must run on a machine with no ML stack at all.

## Primary format: zarr (`s2_adhesion.io.zarr_store`)

### Layout

```
image.ome.zarr/
    0                 array, CZYX, any dtype (whatever the source produced)
    manifest.json     ImageArtifactManifest, canonical JSON
    _SUCCESS           empty marker file, written last

labels.ome.zarr/
    labels/cells/0     array, ZYX, uint32
    labels/nuclei/0    optional, same shape, uint32
    manifest.json      LabelArtifactManifest, canonical JSON
    _SUCCESS           empty marker file, written last
```

`0` is the zarr v3 convention for "resolution level 0" of an OME-NGFF-style
image; this pipeline only ever writes one resolution level. `labels/<name>/0`
mirrors the OME-NGFF labels convention, so `cells` and `nuclei` are separate
label arrays sharing one artifact rather than two channels of one array
(labels can't share integer ids meaningfully the way image channels share a
coordinate grid).

### Manifest fields

Both manifests carry, verbatim, everything needed to reconstruct the
matching `s2_adhesion.contracts` dataclass without guessing:

- **Identity**: `dataset_id`, `field_id`, `source_uri`, `source_field_index`,
  `image_content_sha256` -- reconstructs `FieldIdentity`.
- **Geometry**: `axes`, `shape`, `dtype`, `spacing_um_zyx`, `origin_um_zyx`.
- **Content**: `array_sha256` (top-level; for labels this is the `cells`
  hash), plus a `labels` list of per-array entries
  (`{"name", "present", "array_sha256"}`) covering `cells` and `nuclei`
  independently.
- **Image manifests only**: `channels`, one entry per
  `ChannelBinding` (`channel_id`, `source_index`, `roles`, `source_name`).
- **Label manifests only**: `input_image_artifact_id` (nullable -- the
  producing run may not know the image's artifact id), `input_image_sha256`,
  and `segmentation_provenance` (the full `SegmentationProvenance`: run id,
  backend, strategy, config hash, device, host platform, package/model
  identity).
- `artifact_id` and `created_utc` on both.

Manifests are written with `s2_adhesion.io.manifests.canonical_json`: sorted
keys, fixed indentation, ascii-only. Two writers given the same content
therefore produce byte-identical `manifest.json` files.

### Atomic write / `_SUCCESS` rule

A writer never mutates the destination path in place. It builds the entire
artifact -- arrays, computed hashes, `manifest.json`, and finally
`_SUCCESS` -- inside a temporary sibling directory
(`.<name>.tmp-<random>`), and only then swaps it into the final path.

On POSIX this swap would be one atomic `rename`. **Windows cannot
atomically replace a non-empty directory the same way**: `os.replace` on a
directory requires the destination to not already exist (or to be empty).
`zarr_store._replace_dir` works around this by moving any existing artifact
aside to a backup path first, swapping the new artifact into place, and only
then deleting the backup -- so the final path is always either absent,
holding the complete previous artifact, or holding the complete new one,
never a partial mix, even though it costs two renames instead of one.

Readers additionally **refuse any artifact missing `_SUCCESS`**, regardless
of how it got that way -- an interrupted write, a partial network copy, a
sync tool that doesn't preserve directory atomicity. `_SUCCESS` existing is
the one fact a reader trusts before touching anything else.

### Verification on read

`read_image_volume(path, verify_hashes=True)` and
`read_label_volume(path, verify_hashes=True, image=None)`:

1. Reject if `_SUCCESS` is missing.
2. Load and schema-validate `manifest.json`.
3. Load the array(s); check shape/dtype against the manifest.
4. If `verify_hashes` (the default): recompute each array's sha256 and
   compare against the manifest. A mismatch means the artifact is
   corrupted -- rejected with `ArtifactError`, not silently measured.
   `verify_hashes=False` is a fast path for trusted round trips (e.g.
   same-process tests) and does skip corruption detection.
5. If a `LabelVolume` reader is given `image=`, it calls
   `check_label_image_binding(label, image)`, which independently checks
   shape, spacing, and `input_image_sha256` against the image, raising
   `ArtifactBindingError` (never a bare `ContractViolation`) on the first
   mismatch found. This is the guard that stops labels from one
   preprocessing run being measured against a different image.

Label ids are **never renumbered** at any point in this path -- whatever
non-consecutive ids (e.g. `{1, 5, 900}`) were written come back unrenumbered.

### Chunking and compression

`chunks` (CZYX for images, ZYX for labels) is passed straight through to
zarr and honoured exactly -- readers can independently confirm via
`zarr.open_group(path)["0"].chunks`. Compression is `zstd` via
`zarr.codecs.BloscCodec`, with `compression_level` mapped to `clevel`.

## Interchange format: OME-TIFF (`s2_adhesion.io.tiff_store`)

TIFF is for handing a label volume to tools outside this pipeline (Fiji,
napari, ad-hoc scripts) -- it is not the primary format because TIFF alone
cannot carry everything a label artifact needs:

- **TIFF resolution tags cannot express an anisotropic z spacing or a
  physical origin.** A confocal stack sampled at `(0.5, 0.1, 0.1) um` has no
  faithful single-number TIFF resolution tag.
- TIFF has no field for field/dataset identity, content hashes, or
  segmentation provenance.

So spacing inferred from TIFF tags alone is **never sufficient**, and every
OME-TIFF label artifact this module writes carries a **mandatory** JSON
sidecar that is authoritative for all of that:

```
<field_id>.ome.tiff                    OME-TIFF, series "cells" (ZYX, uint32),
                                        optional series "nuclei"
<field_id>.labels.manifest.json        LabelArtifactManifest, canonical JSON
                                        (same schema as the zarr manifest)
```

`sidecar_path_for(tiff_path)` derives the sidecar name by stripping a
`.ome.tiff` / `.ome.tif` / `.tiff` / `.tif` suffix and appending
`.labels.manifest.json` -- name TIFF artifacts after their `field_id` so this
matches the `<field_id>.labels.manifest.json` convention exactly.

`write_label_tiff` writes the TIFF file first and the sidecar last: an
interrupted write is always observable as "sidecar missing," the same
rejection path `read_label_tiff` already takes for any TIFF that never had a
sidecar at all (e.g. one produced by a different tool). **A TIFF without its
sidecar is rejected outright** -- there is no fallback to TIFF-tag-inferred
geometry.

`read_label_tiff(path, verify_hashes=True, image=None)` otherwise mirrors
`read_label_volume`: schema-validates the sidecar, verifies content hashes
per series, and -- given `image=` -- runs the same
`check_label_image_binding` guard.

## Error types

All artifact-layer failures come from `s2_adhesion.errors`, never a bare
`ValueError`:

- `ArtifactError` -- the artifact is missing, incomplete, malformed, or
  fails a hash check. Covers: no `_SUCCESS`, missing/invalid manifest,
  shape/dtype mismatch against the manifest, content hash mismatch, missing
  TIFF sidecar, non-`uint32` label array.
- `ArtifactBindingError` (subclass of `ArtifactError`) -- the artifact is
  internally fine but does not match the image it's being measured against:
  shape mismatch, spacing mismatch, or image content hash mismatch, each
  raised independently so the caller can tell which one failed.

## Non-goals / things this module deliberately does not do

- No multi-resolution pyramid -- only resolution level `0` is ever written.
- No renumbering, relabeling, or consolidation of label ids, ever.
- No attempt to make TIFF geometry self-sufficient -- the sidecar is
  intentionally the single source of truth rather than duplicating spacing
  into TIFF resolution tags and reconciling the two on read.
