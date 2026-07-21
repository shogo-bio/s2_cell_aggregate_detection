# s2_cell_aggregate_detection

Quantification of *Drosophila* S2 cell adhesion from Nikon `.nd2` confocal stacks.

Two pipelines live here:

- **`s2_adhesion`** — 3D instance segmentation (Cellpose) with per-cell and
  per-aggregate measurements including cell–cell contacts. This is the current work.
- **`detect_aggregations.py`** — the original 2D threshold pipeline, preserved so
  results collected with it stay reproducible and comparable.

**Before using any output, read [`docs/metric_interpretation.md`](docs/metric_interpretation.md).**
It records which measurements survived validation against analytic ground truth and
which did not. Contact area in particular is *not* a primary quantitative endpoint
at the sampling this data was acquired with.

---

## Setup

Python environments for this machine live outside the repo, under `dev/envs`, and
must be **x64** — Cellpose does not run on native ARM64:

```bash
mkdir -p ~/dev/envs/s2-aggregate-x64 && cd ~/dev/envs/s2-aggregate-x64
echo "cpython-3.12.13-windows-x86_64-none" > .python-version   # full identifier, not "3.12"
uv venv --python cpython-3.12.13-windows-x86_64-none .venv
uv pip install --python .venv -e /path/to/s2_cell_aggregate_detection[nd2,cellpose3,dev]
```

Verify you really got x64 — `platform.machine()` reports the *host* and will
mislead you here:

```bash
python -c "import sysconfig; print(sysconfig.get_platform())"   # want: win-amd64
```

See [`docs/cellpose_cpu_notes.md`](docs/cellpose_cpu_notes.md) for why Cellpose 3
(`cyto3`) is the local default and Cellpose 4 (`cpsam`) is not.

**StarDist** is available as an alternative segmentation backend — plausibly a
good fit here, since the transmembrane signal forms a shell and its star-convex
shape prior closes the incomplete rings this biology produces. It needs its own
TensorFlow environment; see [`docs/stardist_notes.md`](docs/stardist_notes.md)
and `configs/example_stardist.yaml`. Both backends implement the same interface,
so you can segment the same extracted artifacts with each and compare.

---

## The 3D pipeline

### Start here: what do the channels actually stain?

The original code assigned the three channels display colours and nothing more.
Nothing in an `.nd2` file records what a dye was, and **no measurement involving
nuclei, organelles or localization means anything until the roles are confirmed
against the experimental record.**

```bash
s2-adhesion inspect data/experiment.nd2
```

prints axes and sizes, voxel spacing in x/y/z, the Z-to-XY anisotropy ratio (feed
this to Cellpose as `anisotropy`), per-channel names and wavelengths where the
file provides them, objective magnification and NA, and a per-Z intensity profile.

Then copy `configs/example.yaml` and set the channel roles. Every role in the
shipped example is a **placeholder**.

### Running it

```bash
s2-adhesion run --config configs/my_experiment.yaml data/experiment.nd2 output/
```

Segmentation is slow on CPU, so the stages are separable and communicate only
through artifacts on disk. This is what lets segmentation run on a GPU machine
while measurement runs anywhere:

```bash
# on the acquisition machine
s2-adhesion extract --config cfg.yaml data/experiment.nd2 artifacts/

# on a GPU machine (needs cellpose + torch)
s2-adhesion segment --config cfg.yaml artifacts/ labels/

# anywhere, including a laptop with no ML stack installed at all
s2-adhesion measure --config cfg.yaml labels/ results/ --images artifacts/
```

`measure` never imports torch or cellpose. That is enforced by tests, not just
convention. It also runs without an image artifact, in which case geometry and
contact metrics are produced and every intensity/localization field is null with a
recorded reason.

### Outputs

| File | One row per |
|---|---|
| `objects.csv` | cell — volume, surface area, sphericity, axes, per-channel intensity, QC flags |
| `contacts.csv` | touching cell pair — contact area, interface orientation, reliability, stability |
| `aggregates.csv` | connected component — cell count, packing fraction, coordination, hull metrics |
| `localization_profiles.csv` | cell × channel × reference × signed-distance bin |
| `field_summary.csv` | field — adhesion mixing index and homotypic/heterotypic contact counts (when populations are configured) |
| `metrics_manifest.json` | every column, with its unit, dtype, nullability and definition |
| `run_manifest.json` | config hash, versions, inputs, outputs, warnings, completion status |

Objects cropped by the edge of the volume keep their `*_observed` values but have
canonical geometry nulled — an object cut off by the field of view has no
meaningful volume, and a zero there would silently bias any average.

---

## The original 2D pipeline

Unchanged in behaviour, including the parts that are scientifically questionable —
changing them would invalidate comparison against already-collected results.

```bash
python detect_aggregations.py input.nd2 output/ --debug
```

| Option | Default | Meaning |
|---|---|---|
| `--s2-diameter` | `10.0` | S2 cell diameter (µm), sets the area threshold and median kernel |
| `--min-cells` | `3` | minimum cell-equivalents to call a region an aggregation |
| `--morph-close-radius` | `1.65` | closing disk radius (µm) |
| `--binary-threshold` | `50` | fixed threshold after median filtering (0–255) |
| `--min-active-channels` | `2` | channels that must be positive in the centre Z slice |
| `--debug` | off | also write per-channel and per-stage intermediate images |

Its CSV columns are `field_id, aggregation_id, area_px, area_um2, active_channels`.

Known limitations, preserved deliberately: it collapses the Z stack to a maximum
projection and so measures occupied *area*, not volume, and cannot count cells;
channel activity is judged from the centre Z slice alone while segmentation uses
the whole projection; and it reads only the x voxel size, ignoring Z spacing
entirely. The 3D pipeline exists to address these.

## Plotting

`tools/boxplot.py` compares `area_um2` across condition folders. It predates this
work and is unchanged; note it uses Student's t-test on area data that is
typically right-skewed.

## Tests

```bash
pytest                      # default run, no ML needed
pytest -m "slow or needs_ml"   # cellpose smoke tests
```

Geometric metrics are validated against closed-form analytic answers on synthetic
volumes rather than against real data, which is stricter — a sphere's volume and a
contact disc's area are known exactly.
