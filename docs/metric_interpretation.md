# How to interpret these measurements

Read this before quoting any number from `objects.csv`, `contacts.csv` or
`aggregates.csv` in a figure or a paper.

Every figure below was measured against closed-form analytic ground truth on
synthetic volumes — two spheres of radius 5 µm whose centres are 8 µm apart,
partitioned by their perpendicular bisector, so the true contact interface is a
flat disc of exactly `pi * (r^2 - (d/2)^2)` = 28.27 µm². Errors are therefore
measurement errors against a known answer, not estimates.

## The short version

At the acquisition sampling this project was designed around —
`(dz, dy, dx) = (0.5, 0.1, 0.1)` µm, i.e. **five times coarser along the optical
axis than in-plane** — measurements split cleanly into two groups.

| Quantity | Error at 5× anisotropy | Use it as |
|---|---|---|
| Volume | 0.4% | a quantitative endpoint |
| Equivalent sphere diameter | 0.1% | a quantitative endpoint |
| Principal axis lengths | sphere gives 10.01 / 10.01 / 9.94 | a quantitative endpoint |
| Packing fraction | 3.7% aniso-vs-iso | a quantitative endpoint |
| Surface area | 3.3% (but ~6.8% below r = 3 µm) | with care; check object size |
| **Contact area** | **up to 45.9%, orientation-dependent** | **orientation-conditioned only** |
| Localization profiles | 5–8% for thin shells on small objects | with care; check shell width |

Anything that depends on **volume** or on **counting** is reliable. Anything that
depends on reconstructing a **surface** degrades, and contact area — which is a
surface between two objects, the hardest case — degrades most.

## Why contact area is demoted

Contact area was the metric this project was built to obtain, because it is the
most direct physical observable of adhesion. It did not survive validation as a
primary endpoint.

Measured error of the default estimator (marching cubes with isotropic
resampling) against the analytic disc, as a function of how the contact plane
happens to sit relative to the imaging axes. Orientation is given by the
interface *normal* — a normal along Z means the contact plane lies flat in XY:

| Interface normal | Angle to optical axis | Error |
|---|---|---|
| Along Z (contact lies flat in XY) | 0° | **4.5%** |
| Along X | 90° | 13.5% |
| Along Y | 90° | 13.4% |
| 45° within the XY plane | 90° | 15.9% |
| Equal parts X, Y and Z | 55° | **42.8%** |

**The error is not monotonic in the angle** — it peaks in the middle, around
55°, and both extremes beat it. That is the body-diagonal direction, where the
interface is maximally misaligned with every voxel face and the staircase bias
is largest.

Two effects compose here. How well the normal aligns with any grid axis governs
the staircase bias; how finely the interface *plane itself* is sampled governs
the rest. A normal along Z puts the contact in the finely sampled XY plane
(0.1 × 0.1 µm), while a normal along X puts it in YZ, sampled 0.1 × 0.5 µm.
That is why along-Z (4.5%) beats along-X (13.5%) even though both are perfectly
axis-aligned.

Isotropic resampling roughly halves the raw error, and it is on by default, but
it cannot recover axial information the microscope never sampled.

The decisive problem is not the size of the error but its **structure**. The bias
depends on each contact's orientation. Cells in an aggregate adhere at arbitrary
orientations, so the bias varies from contact to contact, and it is confounded
with how the cells are packed. **It therefore does not cancel when you compare
two experimental conditions** — if a treatment changes packing geometry, it also
changes the orientation distribution, and thus changes the measurement bias
along with the biology.

For the same reason, `contact_surface_fraction` (contact area ÷ cell surface
area) is *not* a safe way to normalise the problem away. Both numerator and
denominator carry orientation bias, but contact patches and whole-cell surfaces
have different normal distributions, so the two biases do not cancel.

## What to use instead

**Mean coordination number, at the aggregate level.** It asks whether a contact
*exists* rather than reconstructing its area, which removes the orientation
sensitivity. Report the degree distribution alongside the mean, and stratify by
aggregate cell count.

**Packing fraction** (aggregate volume ÷ convex hull volume) as an orthogonal
compaction endpoint. It is volumetric, so it inherits the accuracy of volume.

### The catch with coordination number

Coordination number is not simply "the robust one". It has **binary topological
instability**: a single-voxel bridge invents an entire network edge, and a
single-voxel gap destroys one. The broad axial PSF makes both failure modes
especially likely along Z.

So anisotropy does not stop hurting — its effect changes from a graded bias into
a discrete edge error, which is arguably worse because nothing in the output
looks wrong. This is why every contact carries stability fields recording
whether it survives one-voxel erosion and dilation, and why a
`coordination_number_stable` variant is reported alongside the raw count. **A
coordination number quoted without its stability companion is not evidence.**

## Reliability metadata, and why there is no bias correction

Each contact record carries the area-weighted distribution of its interface
normal angles to the optical axis, how many axial planes support it, a planarity
residual, and a categorical `contact_area_reliability` flag.

The flag's thresholds come from the measured table above, not from intuition.
`high` means the normal is within 20° of Z (the 4.5% regime), `medium` means it
is beyond 75° (the 13–16% regime), and `low` is everything between — the
diagonal band where the 42.8% worst case sits. An earlier version graded
monotonically in angle and therefore ranked the 42.8% case *above* the 13.5%
one; since this is precisely the field a user would filter or weight on, an
inverted flag is worse than no flag at all.

There is deliberately **no orientation-based correction factor** that rescales
area by angle. Such a correction would have to be calibrated, and a calibration
built from a handful of synthetic sphere orientations does not span contact size,
curvature, sub-voxel phase, PSF, noise, or segmentation error. Applying it would
not remove the model error; it would launder it into a number that looks
authoritative. Emitting the orientation metadata is the honest stopping point.

The reliability flag is categorical rather than a continuous predicted error, for
the same reason: three validated orientations cannot support a continuous
calibration curve, and presenting one would imply precision that does not exist.

**Do not filter out low-reliability contacts before comparing conditions.**
Stratify instead. A condition that changes packing geometry changes which
contacts get excluded, so filtering introduces selection bias — exactly the
artefact you would then be tempted to interpret biologically.

## What would actually fix this

Acquire with a finer Z step. At `dz ≈ 0.20 µm` the optical signal is adequately
sampled: with a 0.5–0.8 µm axial PSF that gives roughly 2.5–4 samples per
resolution element, against 1–1.6 at the current 0.5 µm.

But note where that stops helping. **Below about 0.20 µm the PSF, not the
sampling, becomes the binding constraint.** Smaller voxels cannot restore axial
spatial frequencies the optics never transmitted, and interpolation certainly
cannot. If contact area must be the headline endpoint, the answer is better axial
*resolution* or multi-view acquisition, not merely smaller voxels.

Finer sampling also does not by itself make contact area trustworthy — it makes
it worth re-validating. Repeat the orientation validation with a measured PSF and
the full segmentation pipeline before treating the improved numbers as sound.

## A limitation that is not about optics

Every biological interpretation here assumes the channel roles in your config are
correct. Nothing in the code can verify that a channel labelled `nucleus`
actually stains nuclei — it will happily seed a watershed from whatever you point
it at and produce confident, well-formed, meaningless instances. Confirm roles
against the experimental record. `s2-adhesion inspect <file.nd2>` prints the
acquisition metadata to help, but the file does not record what the dyes were.
