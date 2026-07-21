# StarDist backend — setup and what to expect

StarDist is available as an alternative segmentation backend alongside Cellpose.
Both implement the same `SegmentationBackend` protocol, so you can run them on
the same extracted image artifacts and compare the results directly.

## Why it is worth trying here

The labelled proteins are mostly transmembrane, so their signal forms a shell
around each cell. That suits StarDist unusually well: the bright ring **is** the
boundary the model predicts ray distances to.

It should also cope better with the incomplete rings this biology produces. The
protein is not expressed over the whole surface, so the shell has gaps. A
star-convex shape prior closes the outline across a gap; a flow field can leak
through it and merge two cells. S2 cells are near-spherical, so star-convexity
costs little here.

This is a hypothesis worth testing on real data, not a settled result.

## Environment

StarDist is TensorFlow-based; Cellpose is PyTorch-based. They are kept in
separate environments so their numpy/CUDA constraints cannot collide:

```
dev/envs/s2-aggregate-x64   cellpose 4.2.1.1, torch 2.13.0   (main test env)
dev/envs/cellpose-x64       cellpose 3.1.1.3                 (the local default)
dev/envs/stardist-x64       stardist 0.9.2, tensorflow 2.21.0
```

To recreate the StarDist one:

```bash
mkdir -p ~/dev/envs/stardist-x64 && cd ~/dev/envs/stardist-x64
echo "cpython-3.12.13-windows-x86_64-none" > .python-version   # full identifier, not "3.12"
uv venv --python cpython-3.12.13-windows-x86_64-none .venv
uv pip install --python .venv stardist tensorflow pyyaml zarr pandas scikit-image tifffile pytest
.venv/Scripts/python.exe -c "import sysconfig; print(sysconfig.get_platform())"   # want win-amd64
```

Everything on this host runs x64 CPython under emulation. `platform.machine()`
reports the ARM64 host and will mislead you — use `sysconfig.get_platform()`.

## Two things that will bite you

### There is only one pretrained 3D model, and it is a nuclei demo

`StarDist3D.from_pretrained()` registers exactly one model: `3D_demo`. It was
trained on nuclei at anisotropy `(2, 1, 1)`.

Running it on membrane-labelled cells is a **plumbing check, not an analysis**.
Measured on a synthetic three-cell volume with incomplete shells, it produced 41
raw instances for 3 real cells. The backend warns about this on every run and
records `is_pretrained` in provenance, so the fact reaches the results rather
than being forgotten between the run and the figure.

Real use needs a trained or fine-tuned model:

```yaml
model:
  custom_model_dir: ../models/stardist
  custom_model_name: s2_membrane_3d
```

Naming both a pretrained and a custom model, or neither, is a config error.
There is deliberately no silent fallback — quietly running a nuclear model on
membrane data would produce confident, well-formed, wrong instances.

### Anisotropy is baked into the model, not passed at predict time

This is the substantive difference from Cellpose. `CellposeModel.eval()` takes
an `anisotropy` argument; `StarDist3D.predict_instances()` does **not**.
Anisotropy is a training parameter (`Config3D(anisotropy=...)`), so a model
expects input sampled the way its training data was.

There are two ways to be consistent:

- `resample_isotropic: true` (default) — resample the volume to isotropic before
  inference and map labels back to the acquired grid afterwards. Correct for a
  model trained on isotropic data.
- `resample_isotropic: false` — feed the acquired sampling directly. Correct for
  a model trained at this experiment's own anisotropy.

The backend compares the model's recorded anisotropy against what it is being
fed and warns on a mismatch, because nothing else would: StarDist will happily
predict on wrongly-sampled input and return plausible-looking instances.

Note that the default config combination is itself mismatched — `3D_demo` was
trained at `(2, 1, 1)`, not isotropic — and the warning fires accordingly. That
is intended: the default should be right for a properly trained model, and loud
about the demo one.

## A Windows quirk, handled automatically

On Windows without Developer Mode, `from_pretrained` fails with

```
OSError: [WinError 1314] A required privilege is not held by the client
```

csbdeep downloads and extracts the model fine, then fails creating a symlink to
it. The weights are already on disk at that point, so the backend falls back to
loading them directly from `~/.keras/models/StarDist3D/<name>/<name>_extracted/`.
You do not need to enable Developer Mode or run as Administrator.

## Running it

```bash
# extract once
s2-adhesion extract --config configs/example.yaml data.nd2 artifacts/

# segment with each backend
s2-adhesion segment --config configs/example.yaml          artifacts/ labels_cellpose/
s2-adhesion segment --config configs/example_stardist.yaml artifacts/ labels_stardist/

# measure both with the SAME measurement config, so only segmentation differs
s2-adhesion measure --config configs/example.yaml labels_cellpose/ out_cellpose/ --images artifacts/
s2-adhesion measure --config configs/example.yaml labels_stardist/ out_stardist/ --images artifacts/
```

Measurement needs neither TensorFlow nor PyTorch, so the comparison step runs
anywhere.

## Testing

Unit tests inject a fake engine and need no ML installed at all —
`tests/unit/test_stardist.py` runs in the default suite. The one test that
loads the real library is marked `slow` and `needs_ml`:

```bash
# in the stardist environment
pytest tests/unit/test_stardist.py -m "needs_ml"
```

## What is still open

Whether StarDist actually beats Cellpose on this data is unanswered and cannot
be answered without real images and a manually annotated 3D validation set.
Until then, treat backend choice as an open experiment: run both, compare, and
record which produced any given label volume — the provenance carried in every
label artifact already does that for you.
