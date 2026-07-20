# Cellpose on this host: CPU, ARM64, and the v3/v4 split

Audience: someone who has to run cellpose-based segmentation on this
development machine next month and has none of this context yet. Nothing
here is inferred -- every number is a direct measurement on this exact host,
dated, with exact package versions. Where something is genuinely unknown,
that is stated plainly rather than guessed at.

Local environments referenced throughout:

| Env | Path | cellpose |
|---|---|---|
| main | `C:/Users/ryuga/dev/envs/s2-aggregate-x64/.venv` | 4.2.1.1 (`cpsam`) |
| cellpose-3 | `C:/Users/ryuga/dev/envs/cellpose-x64/.venv` | 3.1.1.3 (`cyto3`) |

The project's own test suite (`pytest`, ~330 tests as of 2026-07-18) runs
against the **main** env. `tests/integration/test_cellpose3_cpu_smoke.py`
exercises the cellpose-3 path but is marked `@pytest.mark.slow` and
`@pytest.mark.needs_ml`, both excluded from the default run
(`addopts = "-m 'not slow and not needs_ml'"` in `pyproject.toml`) -- it only
does real work when explicitly selected (e.g.
`pytest -m "slow and needs_ml"`), and even then it skips cleanly on the main
env because the installed cellpose major (4) doesn't match what it needs (3).

## 1. ARM64 host, x64 Python -- and why `platform.machine()` lies to you

This machine (`Windows 11 ARM64`, build 10.0.26200) runs every Python
environment referenced here as **x64 CPython under Windows' x64 emulation
layer**, not native ARM64 Python. This matters because packages with no
ARM64 wheels (cellpose's dependency chain, in particular `torch`) install
and run fine this way, but it is easy to convince yourself otherwise by
checking the wrong thing.

**Do not use `platform.machine()` to detect this.** It reads a host/session
property (effectively `PROCESSOR_ARCHITECTURE`), not the actual Python
build, and on this host it reports `"ARM64"` even inside a pure x64
interpreter -- actively misleading. Verified on the main env, 2026-07-18:

```
>>> import platform, sysconfig
>>> platform.machine()
'ARM64'                 # <-- wrong picture: this is the HOST, not the build
>>> sysconfig.get_platform()
'win-amd64'              # <-- this is the actual Python build: x64
>>> platform.python_version()
'3.12.13'
```

**Use `sysconfig.get_platform() == "win-amd64"`** (or equivalently check
`struct.calcsize("P") == 8` plus `sys.winver`/wheel tags if you need it
programmatically) to confirm an environment is really x64 before trusting
that an x64-only wheel will work in it. `platform.machine()` and
`platform.architecture()` are both host-derived on this OS and cannot be
used for this check.

There is no GPU on this machine. Every timing number below is CPU-only.

## 2. Measured performance

All numbers below are real measurements on this host (Windows 11 ARM64,
x64 CPython 3.12.13 under emulation, CPU only), not estimates.
**Do not re-run the cellpose-4 rows** -- the 2D run alone cost over ten
minutes; re-measuring it burns real wall-clock time for no new information.

| Date | cellpose | Model | Input | Wall time | Result |
|---|---|---|---|---|---|
| 2026-07-18 | 3.1.1.3 | `cyto3` (26 MB U-Net) | 2D 256x256 | 2.2 s | 23 labels, correct |
| 2026-07-18 | 3.1.1.3 | `cyto3` (26 MB U-Net) | 3D 10x256x256 | 128.5 s | 99 labels |
| 2026-07-18 | 4.2.1.1 | `cpsam` (1.23 GB SAM/transformer) | 2D 256x256 | 638.4 s | 20 labels |
| 2026-07-18 | 4.2.1.1 | `cpsam` (1.23 GB SAM/transformer) | 3D 12x48x48 | did not finish in 10 min | pure compute; weights already cached |

Takeaways:

* `cpsam` is **roughly 290x slower per plane** than `cyto3` on this CPU
  (638.4 s vs 2.2 s for the same 256x256 2D plane).
* Local 3D `cpsam` is **operationally nonviable** on this machine: a volume
  smaller than a typical single field of view (12x48x48) did not complete
  in ten minutes, with weights already resident -- that is pure model
  compute, not a download or cold-start cost.
* `cyto3`'s 3D cost does not scale simply with voxel count. cellpose's
  `do_3D=True` path runs 2D inference across the volume from multiple
  orthogonal directions and stitches the result, so wall time tracks
  something closer to "how many 2D passes across all three axes" than raw
  voxel count -- this is why a small-XY/tall-Z volume and a large-XY/short-Z
  volume with the same voxel count will NOT cost the same. Treat any
  extrapolation from a single data point (section 4) as a rough estimate,
  not a guarantee.

## 3. Why `cyto3` is the local default -- an operational choice, not a quality claim

`CellposeModelConfig`'s default (`src/s2_adhesion/config.py`) is
`package_major=3`, `model_name="cyto3"`, `device="cpu"`. This is forced by
the runtime numbers in section 2 and nothing else: `cpsam` at ~290x slower
per plane, and not finishing a 3D volume smaller than a real field of view
in ten minutes, makes it unusable for iterative local development on this
CPU-only host. That is the entire justification.

**This is explicitly not a claim that `cyto3` produces better segmentations
than `cpsam`.** `cpsam` is cellpose's newer SAM/transformer-based model and
plausibly segments crowded, irregular, weak-boundary structures --
exactly the profile of the S2 aggregates this pipeline targets -- more
accurately than a 26 MB U-Net trained on a narrower distribution of shapes.
**This is a real, currently unresolved quality risk**, not a settled
question: nobody has run both models against real S2 aggregate stacks with
manual annotation and compared, and this document does not claim to know
the answer. Before trusting `cyto3` output for anything scientifically
load-bearing on dense/touching aggregates, that comparison needs to happen
on a GPU machine where `cpsam` is actually fast enough to iterate with. If
you're reading this because segmentation quality looks suspect on crowded
regions, this is the first thing to revisit -- it may well be the model
choice, not a bug.

## 4. Extrapolating to a realistic acquisition

A plausible real multi-field nd2 field of view: **512x512 pixels in XY,
20 Z-planes**, `cyto3`, this host, CPU.

Voxel-count ratio against the one measured 3D data point (10x256x256,
128.5 s): `(512*512*20) / (10*256*256) = 8.0x`. A naive linear-in-voxel-count
extrapolation gives **~1030 s (~17 minutes) for one field**.

Treat that as the low end of a rough range, not a committed number --
section 2 already flags that 3D cost likely tracks "orthogonal 2D pass
count" more than raw voxel count, and a 512x512x20 volume has a very
different aspect ratio (much larger XY planes, proportionally fewer Z
passes) than the 10x256x256 volume the estimate is anchored to. With only
one real 3D measurement to extrapolate from, **treat anything in roughly the
15-35 minute range per field as plausible**, and do not commit to a batch
run across many fields without calibrating first.

**Before running a full acquisition**, do one real timed run at an
intermediate size (e.g. 256x256x20, a straightforward one-axis extrapolation
from the measured 10x256x256 point) on this host, and use that number --
not the estimate above -- to decide whether to wait it out locally or move
to a GPU machine. A multi-field nd2 with even a handful of fields at the
15-35 min/field range is realistically an hours-long local job; the GPU
workflow in section 5 is very likely the right call for anything beyond a
one-off single-field sanity check.

## 5. GPU-machine workflow (segmentation elsewhere, measurement here)

The measurement layer (`s2_adhesion.contracts`, the metrics/QC stages) is
deliberately free of any ML dependency and runs correctly with no
`cellpose`/`torch` installed at all -- it only ever consumes a `LabelVolume`
artifact, never a model. That split is what makes this workflow possible:

1. **Extract here.** Read the raw acquisition (`.nd2` or equivalent) and
   write the preprocessed `ImageVolume` out as the project's image artifact
   (zarr/ome-tiff per `ArtifactConfig`) on this machine. No GPU or ML
   package needed for this step.
2. **Copy the image artifact** to the GPU machine (network share, physical
   transfer, whatever's convenient).
3. **Segment there.** Run the same `s2_adhesion` segmentation stage on the
   GPU machine, pointed at whichever `CellposeModelConfig` makes sense there
   (this is exactly where re-evaluating `cpsam` per section 3 belongs --
   `device="cuda"` and cellpose 4 become genuinely usable once inference
   isn't ~290x slower). Output is a `LabelVolume` artifact.
4. **Copy the label artifact back** to this machine.
5. **Measure here, with no ML installed.** `LabelVolume.validate_against()`
   checks the label artifact's shape/spacing/content hash against the image
   artifact before any measurement runs, so a label artifact produced
   against the wrong image (or a stale one) is caught as a
   `ContractViolation` rather than silently measured.

This machine never needs `cellpose`/`torch` installed for steps 1, 2, 4, 5 --
only a machine actually running step 3 does.

## 6. Creating a correctly pinned x64 env with `uv`

This host has both x64 and ARM64 CPython builds available through `uv`, and
`uv` will happily give you either one depending on exactly how you pin it.
**A bare `3.12` pin is not safe on this machine** -- it does not
unambiguously mean "x64 CPython 3.12"; depending on what's already resolved/
cached it can silently resolve to the native ARM64 build, which then fails
(or silently misbehaves) the moment you try to install `torch` or `cellpose`,
neither of which ships an ARM64 Windows wheel.

**This is not hypothetical on this host**: `envs/cellpose-x64/.python-version`
currently contains the bare pin `3.12` (verified 2026-07-18), while
`envs/s2-aggregate-x64/.python-version` correctly contains the full
`cpython-3.12.13-windows-x86_64-none` identifier. The `cellpose-x64` env
happens to currently be x64 (confirmed via `sysconfig.get_platform()`), but
its pin doesn't *guarantee* that on a re-resolve -- it's an existing
inconsistency on this host, not a hypothetical risk, and worth tightening
the next time that env is touched.

Use the full, unambiguous identifier instead:

```
cpython-3.12.13-windows-x86_64-none
```

To create a new correctly pinned env from scratch:

```powershell
cd path\to\new\env\dir
uv python pin cpython-3.12.13-windows-x86_64-none   # writes .python-version
uv venv                                              # creates .venv from that pin
uv pip install -e "path\to\s2_cell_aggregate_detection[cellpose3]"   # or [cellpose4]
```

Confirm it before trusting it with anything:

```
.venv\Scripts\python.exe -c "import sysconfig; print(sysconfig.get_platform())"
# must print: win-amd64
```

`uv python list --only-installed` on this host currently shows
`cpython-3.12.13-windows-x86_64-none` already fetched in two locations
(`%APPDATA%\uv\python\...` and `~/.local/bin/python3.12.exe`), so pinning it
should not require a fresh download.

One more environment-provisioning gotcha specific to the `cellpose-x64` env
as it exists today: it has `numpy`, `scipy`, `pandas`, `PyYAML`, and
`tifffile` (cellpose's own dependency chain pulled these in), but **no
`pip` and no `pytest`** -- it was set up purely for raw cellpose benchmarking,
not for running this project's test suite. To run
`tests/integration/test_cellpose3_cpu_smoke.py` for real against this env,
add `pytest` (and `zarr`, which is also absent) with
`uv pip install --python <path to that .venv> pytest zarr`, or provision a
fresh env the way section 6 above describes and install with the
`[cellpose3]` extra plus `dev`.

## 7. cellpose 3 vs cellpose 4: API differences a future maintainer will trip over

These are genuine, verified differences between `models.Cellpose.eval`
(v3) and `models.CellposeModel.eval` (v4) -- not naming choices in this
project's adapters. Confirmed 2026-07-18 by inspecting
`inspect.signature(...)` on both installed versions on this host.

* **`z_axis=`**: required (and enforced) on v4's 3D eval -- omitting it
  raises `ValueError`. v3 has no such argument at all; 3D volumes are passed
  channel-last (or bare `(Z, Y, X)` for single-channel) instead.
* **Channel selection**: v3 uses the `channels=[cytoplasm_idx, nucleus_idx]`
  1-based convention (`0` = "not present"). v4 has no such argument --
  multi-channel volumes are passed with `channel_axis=` and cpsam
  auto-detects channel content.
* **Class/entry point**: v3's public API is `models.Cellpose(...).eval(...)`,
  which internally delegates to a `models.CellposeModel` instance (accessed
  as `.cp` in some versions). v4's public API IS `models.CellposeModel(...)
  .eval(...)` directly -- there is no separate wrapper class in v4.
* **Tiling**: v4's real `eval()` signature has no boolean `tile=` switch --
  only `tile_overlap=`/`bsize=`, with tiling effectively always on for large
  images.
* **`tile=` on v3 is ALSO not what it looks like on this host -- this is a
  live, unresolved bug, not just a historical note.** The installed
  cellpose 3.1.1.3's actual `CellposeModel.eval()` signature (verified via
  `inspect.signature`, 2026-07-18) has **no `tile` parameter either** -- it
  takes `tile_overlap=`/`bsize=`, exactly like v4. `models.Cellpose.eval()`
  forwards unrecognised kwargs (including `tile=`) straight through to the
  underlying `CellposeModel.eval()`, so calling it with `tile=` at all --
  regardless of value -- raises:
  ```
  TypeError: CellposeModel.eval() got an unexpected keyword argument 'tile'
  ```
  `s2_adhesion.segmentation.cellpose_v3.CellposeV3Engine.evaluate()`
  currently forwards `tile=eval_cfg.tile` unconditionally (this is a
  different file/workstream than this doc, not something fixed here). The
  practical consequence: **every real 3D `DirectCellposeBackend` run against
  cellpose 3.1.1.3 on this host currently fails with that `TypeError`**,
  confirmed by running the real backend end-to-end against the
  `cellpose-x64` env. `tests/integration/test_cellpose3_cpu_smoke.py`
  records this precisely as a `strict=True` `xfail` (matching exactly on
  `TypeError`) rather than silently working around it or hiding it -- when
  `cellpose_v3.py` stops forwarding an incompatible `tile` kwarg, that test
  will `XPASS` and pytest will flag the run, which is the signal to remove
  the marker. Until then, do not trust a `cyto3` 3D run through this
  project's backend on this host to actually complete.
* **3D diameter auto-estimation isn't available.** Passing `diameter=None`
  for a 3D (`do_3D=True`) eval on v3 does not raise, but silently prints
  `"could not estimate diameter, does not work on non-2D images"` and falls
  back to a default rather than actually estimating from the image
  (observed while validating the smoke test, 2026-07-18). If cell size
  matters for your run, set `diameter_um` explicitly on
  `CellposeModelConfig` for 3D work; do not rely on auto-estimation the way
  you might for a 2D-only workflow.
* Every `CellposeEvalConfig` field is forwarded by name in both adapters
  (never through a generic `**kwargs` filter) specifically so a mismatch
  like the `tile=` one above is a loud, traceable `TypeError` at a known
  line rather than a silently-dropped parameter -- that design is doing its
  job here, even though the underlying bug is still open.

## 8. What is genuinely unknown

Stated plainly, not glossed over:

* **Whether `cyto3` is good enough for real S2 aggregates is unknown.** No
  comparison against `cpsam` (or against manual annotation) has been done on
  real data. Section 3 is not a quality endorsement.
* **The 512x512x20 extrapolation in section 4 is a rough range**, anchored
  to exactly one real 3D measurement point with a different aspect ratio.
  Calibrate with a real intermediate-size run before committing to it.
* **Whether the `tile=` bug in section 7 is the only such incompatibility
  between this project's assumptions and the actually-installed cellpose
  3.1.1.3 API is unknown** -- it was found by trying to run a real smoke
  test, not by an exhaustive audit of every forwarded kwarg.
