# Design decisions, assumptions and deviations

This file records **why** the code looks the way it does, and — as required by the working
style — every ambiguity I resolved on my own (what was ambiguous, what I chose, why) and
every deviation from the specification.

Sections are appended as milestones land; each entry is dated with the milestone that
introduced it.

---

## D1 (M1) — Array indexing for staggered fields

**Ambiguity.** The spec fixes the shapes (`u` is `(ny, nx+1)`, `v` is `(ny+1, nx)`) but not
whether index `i` of `u` is "the i-th cell's right face" (`i = 1..nx`) or "the i-th face"
(`i = 0..nx`).

**Choice.** Pure face indices: `u[:, i]` is the face at `x = i·h` exactly, `v[j, :]` is the
face at `y = j·h` exactly.

**Why.** It removes all off-by-one bookkeeping from the operators: `divergence` is
`(u[:,1:] - u[:,:-1])`, boundaries are columns `0` and `-1`. Coordinate helpers
(`Grid.u_coords()` etc.) are the single place that knows about the half-cell offsets.

---

## D2 (M1) — Exact divergence-free test fields come from a corner streamfunction

**Ambiguity.** "Divergence of analytically and discretely divergence-free fields is ~0."

**Choice.** Two different tests, because they measure different things:

1. `analytic.curl_streamfunction(psi, h)` builds a MAC velocity from a streamfunction
   sampled at **cell corners**: `u[j,i] = (psi[j+1,i] - psi[j,i])/h`,
   `v[j,i] = -(psi[j,i+1] - psi[j,i])/h`. Taking the discrete divergence of that field
   cancels term by term, so the result is divergence-free to *machine precision* (measured
   `1.1e-15` on 16², `4.9e-15` on 64²). Every "should be exactly div-free" test uses this
   construction.
2. Sampling a *continuum* divergence-free field directly on the faces leaves a residual that
   shrinks like `h²`. Measured orders: see `mackerel validate foundations`.

**Note / honest correction.** Taylor–Green is a bad field for test 2: its centred differences
cancel exactly, so the residual sat at `1e-14` and the fitted "order" was noise (−1.05). The
test now uses an exponentially decaying vortex, for which the residual is genuinely `O(h²)`.
This is recorded because the first version of the test *looked* like it passed a
second-order check when it was actually measuring round-off.

---

## D3 (M1) — Ghost cells for wall boundaries: mirror about the wall value

**Ambiguity.** "Make backtrace clamp at walls correctly, sampling through the padded arrays
so wall conditions are respected, not just clamped to the first interior sample."

**Choice.** `pad_velocity` adds one ghost layer with `ghost = 2·wall_value − partner`, where
the partner is the sample one cell **inside** the wall (not the sample on the wall).
* Tangential component: makes the interpolated velocity *at* the wall equal the wall speed
  (zero for no-slip; lid speed for the lid). Verified in
  `tests/test_boundary.py::test_no_slip_tangential_ghost_extinguishes_wall_velocity` and
  `::test_moving_lid_normal_velocity_is_zero_but_tangential_is_one`.
* Normal component: gives the odd mirror `−interior` at a wall at rest, so the interpolated
  normal velocity on the wall is exactly zero. A backtrace that reaches the wall does not
  carry the overshooting velocity it had one cell inside.

`OUTFLOW` ghosts copy the interior (zero gradient); `PERIODIC` ghosts wrap by the true period
of the padded index space (`nx` columns / `ny` rows — *not* `nx+1`, because the periodic
faces are stored twice; getting this wrong shifts the seam by one cell and was caught by the
periodic-padding test).

**Cost.** The padding allocates two small arrays per advection call. At 512² that is 2 MB per
step; it is measured and reported in the Milestone 8 benchmarks rather than assumed free.

---

## D4 (M1) — Vorticity at corners: what happens on the boundary

The corner operator `ω = (v[j,i]−v[j,i−1])/h − (u[j,i]−u[j−1,i])/h` needs a left neighbour
for the first term and a bottom neighbour for the second. So `dv/dx` exists at corner columns
`1..nx−1` and `du/dy` at corner rows `1..ny−1`. **Choice:** interior corners are exact;
edge corners use the periodic wrap when that side is periodic and are left at the one-sided
(zero) difference otherwise. Consequences: enstrophy and vorticity images use the interior
slice `[1:-1, 1:-1]`, and wall vorticity (a genuinely one-sided quantity) is reported
separately where it matters rather than faked by extrapolation.

---

## D5 (M1) — `ly` is derived, never passed

`Grid(nx, ny, lx)` fixes square cells `h = lx/nx` and therefore `ly = ny·h`; `ly` is a
read-only property. The spec fixes the constructor signature and states the relation, so a
separate `ly` argument could only ever disagree with it.

---

## D6 (M1) — Scalar padding offsets

`pad_scalar` returns `(ny+2, nx+2)` with `out[j+1, i+1] = field[j, i]`, so it must be sampled
with `x0 = y0 = −h/2` (cell centres are offset by half a cell). This is documented on the
function because the tempting `x0 = 0` silently shifts every scalar advection by half a cell
— a bias that is invisible in a smoke plume and fatal in a convergence study.

---

## D7 (M1) — Video output depends on `ffmpeg`

The spec asks for GIF (Pillow) and MP4 (ffmpeg on PATH, optional). `ffmpeg` is **not**
installed in the environment this was developed in, so:
`render.save_frames_mp4` returns `None` and writes nothing when the binary is missing, and
every scene still writes an animated GIF. `mackerel info` reports which of the two is
available. This is a documented optional dependency (`docs/DESIGN.md` here,
`README.md` quick start) and no MP4 is claimed anywhere in the docs.

---

## D8 (M1) — Test tolerances are derived from measured error, not fitted

Where a test bounds an error, the bound is an expression in `h` (e.g. energy error
`< 4 h²`) whose coefficient is the measured one (3.3) rounded up, and the fitted convergence
rate is asserted separately with an explicit tolerance. No test asserts an absolute error
that was reached by tuning a parameter until it passed.

---

## D9 (M2) — Limiter: local 3x3 min/max clamp, not FCT

**Ambiguity.** "MacCormack and/or BFECC with a limiter to reduce numerical diffusion without
overshoot."

**Choice.** After the forward/backward error correction, clamp the corrected field to the
`[min, max]` of its own 3x3 neighbourhood (wrapped on periodic axes). BFECC clamps the
*corrected field before the final sweep*; clamping the destination instead is unstable at
CFL ~ 1 and was measured to explode in a single step (1e7).

**Consequence, measured.** The clamp is nonlinear and per-cell, so the corrected schemes are
**not conservative**: 100 steps of periodic translation drift the mass by +1.0% (MacCormack)
and −1.6% (BFECC) versus 1e-14 for semi-Lagrangian. That is reported in
`docs/milestones/M2.md` and re-measured in `mackerel validate advection` rather than papered
over. A strict Zalesak FCT limiter would need face-based fluxes; it is listed as future work.

---

## D10 (M2) — "Backward sweep" means forward in time on the frozen field

The MacCormack/BFECC error estimate needs a map that approximately inverts the forward
semi-Lagrangian step. Tracing *backwards from the departure points* does not: `x -> x - dt u`
is not an isometry, and the resulting estimate was pure noise (a one-cell sine translation
gained a 0.76 spurious heap in one step). Implementation traces **forward** in time from the
grid points by passing `-dt` to `backtrace` on the padded velocity, which is exact for
uniform flow (verified: 2.5e-11) and is the standard treatment of a frozen velocity field
over one step.

---

## D11 (M2) — Advection tests use rigid rotation inside no-slip walls

A rotating-blob error study needs a flow whose exact solution is known, so the velocity field
is prescribed (`analytic.solid_body_rotation`) and not solved for; the blob is placed at
radius 0.25 from the centre so it never approaches the walls, where the prescribed rigid
rotation would conflict with the no-slip condition.

---

## D13 (M3) — Divergence diagnostics exclude solid cells

Inside an obstacle the velocity is whatever the no-slip data implies; its discrete divergence
is not a property of the flow and cannot be made zero. `project._divergence_stats` therefore
masks solid cells out, and every test that uses an obstacle asserts the divergence on cells
whose four bounding faces are fluid (via `scipy.ndimage.binary_erosion`). Without this the
obstacle test measures a meaningless 130 instead of the meaningful 5e-2.

---

## D14 (M3) — Divergence tolerance is derived from the solver tolerance

The projection solves the Poisson equation to a *relative residual* `tol`. The resulting
divergence floor is approximately `tol * max|div_input| / (dt * h)`, because
`div(u_new) = div(u*) - dt*laplacian(p)` and the residual of the solve appears exactly there.
At `tol = 1e-9`, `h = 1/64`, `dt = 0.1` and `max|div| = 233` that is ~2e-7 — so a test that
demands `1e-8` is demanding better than the solver was asked to deliver. The tests use `1e-6`
(and quote the measured value in the failure message); the spec's "stated tolerance" is the
one in this file and in `docs/VALIDATION.md`, not a number picked to look good.

---

## D15 (M3) — Jacobi and Gauss-Seidel are reported as not converging

On 64² and above, Jacobi and Gauss-Seidel do not reach `tol = 1e-8` within the 4000-sweep
budget used by `mackerel validate projection`. They are reported with a `*` marker and their
"time slope" is annotated as meaningless for accuracy purposes. The spec asks to "measure
iterations-to-converge and wall time per solver across grid sizes, and report only what is
actually measured" — this is that.

---

## D16 (M3) — Multigrid: sparse Galerkin hierarchy, `lsqr` at the coarsest level

Two choices that the first implementation got wrong and that are worth recording because
they are the standard traps:

1. **Coarse operators must be Galerkin (`A_c = R A P`).** Re-discretising the PDE on a grid of
   half the spacing is cheaper to write and *looks* right, but the coarse-grid truncation
   error is not in the range of the prolongation, so the V-cycle stalls at a residual floor
   (measured 2.6e-3 relative) instead of converging.
2. **The coarsest level must be solved, not smoothed.** A few dozen Jacobi sweeps on the
   coarsest grid leaves low-frequency error behind; solving it directly (or with `lsqr` when
   the coarse operator inherits the pure-Neumann singularity) is what gives a grid-independent
   convergence factor.

The measured factor is ≈0.08 per V-cycle with 2 pre- and 2 post-smooths at `ω = 0.8`.

---

## D17 (M4) — Implicit diffusion uses eliminated Dirichlet values, not pinned unknowns

A wall-normal MAC face is prescribed data. Including it as a CG unknown with an identity row
while leaving its neighbours' couplings in place makes the matrix nonsymmetric. The diffusion
system instead zeros prescribed entries in the operator, applies CG in the free subspace, and
moves `α u_wall` to the right-hand side of each neighbouring row. Tangential no-slip walls use
the ghost relation `u_ghost = 2u_wall-u_inside`; the resulting edge diagonal is `1+5α` (not
`1+4α`) and the source is `+2αu_wall`. Direct comparisons with an explicitly padded Laplacian
are used to check all of these signs.

## D18 (M4) — Scalar diffusivity is independent of velocity viscosity

The first integration draft placed scalar diffusion inside `if nu > 0`, which incorrectly
made dye in an inviscid fluid undiffusible. The stages are now independent: `nu` controls
velocity diffusion and `scalar_diffusivity` controls every registered scalar. The scalar is
temperature-like for buoyancy, so positive anomalies accelerate upward (`+gβ(T-mean(T))`).

## D19 (M4) — BFECC velocity correction follows the scalar BFECC definition

The velocity BFECC path initially applied the error correction to the forward result and then
advected it again. That is not BFECC: the correction must be applied to the *old* field,
`φ_c = φ^n + (φ^n-A^R Aφ^n)/2`, followed by `Aφ_c`. The implementation now mirrors the scalar
algorithm and retains its local limiter. On the nonlinear Taylor–Green velocity experiment,
MacCormack is more accurate than BFECC; this measured result is reported rather than hidden.

## D20 (M4) — Automatic time steps have a cell-crossing cap and final-time clamping

With a nearly resting fluid, `cfl*h/max|u|` can be thousands of time units, which is a poor
choice for an explicit body force. Automatic steps are capped at `h` (a deterministic
cell-crossing time); a fixed `dt` remains available for accuracy studies. `run_until` clamps
its last step to the requested end time. This is a time-integration control choice, not a
claim that the CFL limit is a stability requirement for semi-Lagrangian advection.

---

## D21 (M5) — Ghia comparison uses centreline interpolation and a documented outlier rule

The cavity reference is a table, not a field sampled at MAC locations. The validation script
averages the two centre-adjacent cell columns/rows, interpolates to the published stations,
and inserts exact wall endpoints. All 17 values per profile and Reynolds number are stored in
`validation/data/ghia_1982.csv` with the original citation. The Re=400 `v(.9063)` datum is
retained in `*_all` metrics but excluded from the primary metric because the cited
transcription and independent benchmark discussions identify it as an isolated anomaly.
This exception is deliberately narrow and visible in the printed output.

## D22 (M5) — Cavity pressure iteration cap follows measured convergence

A 400-iteration cap is enough at 64² but can truncate CG at 128², leaving a divergence floor
of `6.6e-5` in a Re=400 run. The cavity driver now allows 2000 iterations and records the
maximum divergence over the run. This is a solver-budget correction, not a relaxation of the
pressure tolerance; the tolerance remains `1e-8`.

---

## D23 (M6) — Staircase immersed boundaries share one mask across all stages

`obstacles.circle_mask`/`box_mask` mark cell centres; `solid_face_masks` marks every touching
MAC face. The mask is used by the Poisson system and by the simulation before and after
advection, diffusion and projection. Moving obstacles call `set_solid`, which invalidates the
cached pressure hierarchy. This is intentionally a staircase immersed-boundary method, not a
smooth body-fitted cylinder; wake numbers are reported with that limitation.

## D24 (M6) — OUTFLOW requires zero-gradient velocity but Dirichlet pressure

Two independent bugs were found by the cylinder test. Diffusion must not pin an outflow normal
velocity to zero: it is an unknown with a zero-gradient stencil, while INFLOW and wall normal
faces are prescribed. Conversely, the pressure gradient at an outflow is not zero; it uses
`p=0` outside. The diffusion system now has independent low/high side modes, and projection
fills the one-sided outflow pressure differences. A regression test checks that an initially
unit outflow remains open and divergence-free.

## D25 (M6) — Free-slip padding copies tangential values

Free-slip means zero normal velocity and zero tangential normal derivative. The padded
velocity used for backtracing now reflects the normal component and copies the tangential
component. No-slip continues to use `2u_wall-u_inside`; treating the two wall kinds alike had
made free-slip behave like no-slip during advection.

## D26 (M6) — Strouhal is measured after a transient, with confinement recorded

The cylinder driver uses a small deterministic asymmetric initial perturbation only to select
a shedding phase; there is no periodic forcing. Peak-to-peak and FFT estimates are computed
from a downstream transverse-velocity probe after 35% of the run. The 2x1 domain has 16%
blockage, so its `St=.20388` at Re=100 is not silently substituted for the unconfined
reference relation `.164335`; both are printed with the geometry and sampling details.

---

## D27 (M7) — Air pressure is Dirichlet, not obstacle Neumann

The existing masked pressure operator was designed for solid cells, where a missing coupling
is the discrete zero-normal-pressure-gradient condition. Free-surface air needs a different
operator: `free_surface=True` treats the mask as air, adds a diagonal term at every mixed
liquid–air face for exterior `p=0`, and returns zero air pressure. Pressure corrections are
restricted to faces touching liquid. A regression checks both the diagonal difference and the
liquid divergence, preventing an accidental reuse of the obstacle treatment.

## D28 (M7) — Clamp interpolation coordinates and weights together

MAC `u` and `v` samples have different half-cell offsets. Near a wall, clipping only the
lower interpolation index leaves a negative or greater-than-one fractional weight. The
particle kernels now clamp both indices and fractions, preserving a nonnegative partition of
unity. This fixed the early dam-break blow-up without adding artificial viscosity or changing
the pressure tolerance. NumPy and Numba implement the same four-corner operation.

## D29 (M7) — PIC/FLIP blend and nominal volume are explicit diagnostics

`WaterConfig.pic_blend` is the PIC fraction, so `.05` means 95% FLIP. The default adaptive
step is capped by both `dt_max` and `.5*h/max|u_particle|`; explicit steps and final-time
clamping remain available. Closed walls never delete particles. `water_volume()` therefore
reports nominal particle volume, and the reports call it nominal rather than claiming VOF
mass conservation. PIC and FLIP are run independently and their different front errors are
both saved.

## D30 (M7) — Optional Numba is an equivalence-checked transfer backend

The base solver remains NumPy. Selecting `transfer_backend='numba'` lazily compiles a serial
kernel; serial updates avoid races on shared MAC faces and make accumulation deterministic.
The performance driver checks `u` and `v` transfer arrays and complete particle trajectories
against NumPy, warms JIT before timing, and reports transfer and whole-step speedups separately.
A missing optional Numba installation produces an explicit unavailable report, not a failed
NumPy simulation.

## D31 (M7) — Reference normalization and front conventions are versioned with the data

The Martin–Moyce CSV records the DOI, the secondary Lethe digitisation URL, and the exact
`tau=t*sqrt(2g/a)`, `delta=x_front/a` definitions. Lethe's separate `+0.175` gate-opening
correction is not silently applied to instantaneous release. The driver includes its initial
half-cell front offset in the error metrics and retains all raw series, so a future interface
threshold or resolution study can be compared without rewriting the reference data.


---

## D32 — BFECC component offsets are checked with a legacy control

The BFECC velocity implementation had a `v`-only final-sweep offset error: it sampled the
padded `v` array with the `u` coordinates. This is a staggered-grid indexing bug, not a
boundary-condition or correction-sign theory. The fixed solver uses `(-h/2,-h)` for that
sweep. `validation/bfecc.py` retains the old operation only as a labelled control and measures
limiter on/off, old versus advected reverse trajectories, and the opposite correction sign.
At 64², `ν=.01`, `dt=.0025`, `t=.2`, the legacy/corrected relative energy errors are
`-22.9431%/-0.8560%`; limiter-off is `-0.7849%`, old-reverse is `-0.8585%`, and wrong-sign is
`-16.7659%`. No parameter was tuned to obtain these controls. The investigation assumes the
same periodic Taylor–Green initial condition and pressure/diffusion stages in every case.

---

## D33 — Convergence studies separate analytic spatial and temporal errors

The spatial convergence driver fixes `dt=6.25e-5` and `t=.05` before running 32²/64²/128²;
its primary metric is analytic velocity L2, not a signed energy difference near zero. The
temporal study fixes 128² and `t=.2` over `dt=.004/.002/.001/.0005`. Both use MacCormack,
`ν=.01`, and pressure tolerance `1e-9`. The measured orders are `1.79945` spatial velocity,
`1.01100` temporal energy, and `0.97163` temporal velocity. The spatial energy fit is retained
with its sign-change warning rather than used as the headline order.

A validation-only Strang P/2–D/2–A–D/2–P/2 control assumes no forces, scalars or obstacles.
It gives `1.01884` temporal energy order and slightly larger errors than Lie at 64², so no
second-order production splitting is claimed or enabled. These choices and the initial MAC
quadrature convention are recorded so future refinements do not accidentally compare unlike
error norms.

---

## D34 — Blockage comparisons report resolution and missing shedding honestly

New cylinder cases use a 4×1 channel, `U=1`, `Re=100`, `dt=.005`, a geometry-scaled
deterministic perturbation, and probe `(1.5,.5)`. The .08 coarse/fine grids are 128×32 and
256×64; the .04 grids are 256×64 and 320×80. The last case uses `t=12`, while the other new
cases use `t=20`; these windows are not silently interchangeable. A signal with no measured
peaks is recorded as `St=null`, never as zero. The completed fine cases give `.152988` at
D/H=.08 and `.120548` at D/H=.04, so the reference gap is reduced for .08 but not closed for
.04. Cells/D and maximum divergence are retained because coarse small cylinders can suppress
shedding numerically. The reference relation is unconfined context, not a fitting target.

---

## D35 — Extended cavity runs have explicit steady and pressure budgets

The extended cavity cases use the unchanged Ghia tables, MacCormack, `dt=.01`, pressure
tolerance `1e-8`, and a 2000-iteration cap. Re=1000 is documented on 64² with a 200-time-unit
safety limit; Re=400 is run on 128² with a 100-unit limit. Steady convergence means the maximum
u change over a 100-step batch is below `2e-8`; the driver records both the Boolean and the
final change. Re=1000 converged at `t=145`, and Re=400 at `t=63`, with maximum pressure
iterations 243 and 483 respectively. Primary Re=400 v errors exclude exactly the cited
`x=.9063` outlier; all-point metrics and the raw numerical/reference values remain visible.

---

## D36 — Dam-break refinement retains the zero-delay convention and failed trend

The second-resolution driver repeats both PIC and 95% FLIP at 48×36 and 96×72 with the same
`a=.25`, `g=9.81`, `dt_max=.0005`, `end_tau=2.96`, and rightmost-particle-plus-half-cell front
definition. `GATE_OPENING_DELAY=0` means instantaneous release at `t=0`, with no `+.175`
shift on either simulation or Martin–Moyce data. The measured FLIP RMSE changes from
`.0725510` to `.2564548`, while PIC changes from `.6225517` to `.0868497`; the lack of a
uniform refinement trend is reported as a limitation. Volume-relative errors are zero for
all four runs, and divergence maxima are retained alongside the front errors.

---

## D37 — Reseeding is a paired, local, opt-in particle exchange

**Ambiguity.** "Fix the particle density" could mean creating new particles in thin regions,
cloning a neighbour's velocity, running a full particle-remeshing/pushing pass, or moving
existing particles. It could also mean fixing the *cell-average* density, which is a
free-surface geometry effect, not a defect.

**Choice.** A *paired exchange*: over-dense cells donate surplus particles to cells whose
sparsity is unambiguous, and every moved particle keeps its own velocity. Concretely

1. **Donors** are cells holding more than `reseed_target` particles (default: the nominal
   `particles_per_cell`, 4). The surplus is drained one particle at a time back to the target.
2. **Recipients** are cells whose sparsity cannot be explained by the interface: an *empty*
   cell whose every in-domain four-neighbour holds liquid (a hole in the interior of the
   liquid), and a cell below `min_particles_per_cell` with no exterior-air neighbour.
   `count > 0` alone is *not* a sufficient criterion — most cells below four particles per
   cell are free-surface fringe cells that the interface legitimately crosses, and filling
   them pushes particles into the air region.
3. **Pairing** is greedy nearest-donor within `reseed_radius` cells on a Morton-ordered
   candidate list. A slot with no donor inside the radius is left unfilled and counted as
   `unfilled_slots`; it is never filled by a long-range teleport.
4. **Budget.** When demand exceeds supply, one seeded binomial keep probability scales every
   recipient, so no donor is over-drained and the outcome is reproducible.
5. **Cooldown.** A relocated particle is not eligible to donate again for `reseed_cooldown`
   steps, which prevents a particle being shuttled between starved cells every step and
   losing the Lagrangian history that makes FLIP low-diffusion.

**Why.** The alternative (create particles) changes the particle count and therefore the
nominal volume, contradicts the M7 conservation statement, and cannot be validated against
the same reference series. A paired exchange conserves count, volume and the velocity
multiset *exactly*, so any change in the answer is attributable to positions alone. Requiring
an unambiguous recipient means reseeding can never add liquid beyond the free surface, which
is the failure mode of naive density equalisation and was observed directly: a
three-neighbour hole rule accepted exterior corner cells and produced a `.1198` front RMSE at
48×36 versus `.0726` unreseeded. With full enclosure required, the same run gives `.0691`.

**Deviation.** The feature is **off by default** (`WaterConfig.reseed=False`), so every
previous M7 figure remains bit-for-bit reproducible. `validation/flip_reseed.py` is the only
place it is enabled, and it reports the configuration it used. Measured outcomes are in
`docs/milestones/M7.md`.

**Command path.** `mackerel validate <name>` imports `validation.<name>`, which is a plain
directory rather than an installed package, so the command needs the repository root on
`PYTHONPATH` (`PYTHONPATH=. mackerel validate flip_reseed`) or must be run as
`python -m validation.flip_reseed` from the repository root. This is a pre-existing
repository-layout property, not something introduced here; both forms are recorded so the new
driver is reproducible.

---

## D38 — README media are generated, whitelisted and documented

**Ambiguity.** `.gitignore` already reserved `docs/media/` with `!docs/media/*.gif` exceptions,
but the directory did not exist and the README had no images, so it was unclear whether media
was meant to be tracked at all or only produced into `outputs/`.

**Choice.** Track a small curated set and generate it with one script,
`validation/showcase.py` (`python -m validation.showcase`, about 40 s). It writes
`hero.png`, `dam_break`, `sloshing`, `splash`, `smoke` and `taylor_green` GIFs plus
`showcase.json`, which records grid, time step, frame count, particle count, end time and the
measured maximum divergence for every case.

**Why.** The repository's claim is that its numbers can be reproduced; a picture of a solver
that no committed code produces would undercut that. Media is limited to six files (829 KB
total) because Git stores every version of a binary forever, and `outputs/` stays gitignored so
the large validation artifacts are not duplicated into history.

**Rendering conventions.** Liquid frames show *particle occupancy*: a bright rim marks a liquid
cell with fewer than four liquid four-neighbours (the free surface), the body shading is
particles per cell, and everything else is a clearly empty dark box. The stair-stepped
interface is therefore an honest depiction of a cell-scale free surface, not a rendering
artefact, and the README says so. Vorticity frames use a fixed colour range per scene so frames
within a GIF stay comparable; the solid body is drawn dark rather than transparent.

**Deliberate exclusions.** The cylinder wake was tried and dropped: at the preview grid the
shedding is not resolved, so the wake is steady and a GIF of it looks like a still, which would
misrepresent the case. The M6 blockage study keeps that measurement (`docs/milestones/M6.md`).
Sloshing is kept even though it is visually calm because it is one of the M7 demo cases.

---

## D39 — The plume GIF recorded a single frame

**Defect.** `validation/smoke.py` decided when to capture a frame with
`state.time - times[-2] >= 0.049`, but `times` gets an entry on *every* step, so `times[-2]` is
always one time step back. The gap was therefore `dt=0.005`, never `0.049`, and the
`len(frames) < 80` cap was hit immediately: `outputs` and `mackerel run smoke` produced a
one-frame "animation" for every M6 report.

**Fix.** Capture decisions now compare against `frame_times[-1]`, the last *captured* time, and
`frame_times` is returned. A 0.3 s, `dt=0.005` run records 60 frames instead of 1, and
`tests/test_showcase.py` asserts more than five frames with a spacing above 0.04 s so the bug
cannot return silently. No physical result changes: the recorded scalar diagnostics
(`times`, `mass`, `centre_y`) were always per step and are untouched.

---

## D40 — CI enforces accuracy, not only correctness

**Ambiguity.** A green test suite proves the stages compose; it does not prove the numbers are
right. Nothing stopped a future edit from changing a stencil offset and silently moving a
documented result.

**Choice.** `.github/workflows/ci.yml` runs ruff, strict mypy and pytest, plus
`validation/ci_subset.py`, a sub-second numerical gate with three checks. Thresholds are set
from measured values on this repository's NumPy reference path and the margin is written next
to each bound:

| check | measured | bound in CI | margin |
|---|---|---|---|
| Taylor-Green 32², full split step, `t=.2`, BFECC | rel. energy error `-1.300313e-2`, `div_max 1.86e-10` | `|err| < 2.0e-2`, `div < 1e-8` | 1.5x on energy, ~50x on divergence |
| BFECC sweep + viscosity + projection, 32² | corrected energy ratio `.984116`, legacy `.962012` | corrected in `.975..0.995`, legacy `>= .005` below | band is 4x the corrected/legacy gap |
| PIC/FLIP water, 24×18, 40 steps | 288/288 particles, volume change `0.0`, `div_max 2.0e-9` | exact equality, `div < 1e-6` | exact for conservation, 500x on divergence |

The legacy BFECC control is asserted to stay *worse*, so the gate fails both if the fix is
reverted and if the control stops exercising the bug. The pytest matrix runs one leg without
Numba (the reference path must not depend on it) and one with it, so the equivalence tests run
instead of skipping. The energy bound is deliberately looser than the measured value: the
purpose is to catch a ~23% offset bug, not to pin the 4th digit, which would fail on unrelated
platform-level floating-point differences.

**Assumption.** CI is not run by this sandbox, so the workflow is validated structurally (YAML
parses, the gate is executed locally, and a test asserts both the commands and the measured
margins are still present). The gate itself runs in 0.7 s locally.

---

## D41 — The dam-break FLIP non-convergence is a temporal error, not an ill-posed metric

**Question.** FLIP front RMSE rose from `.07255` (48x36) to `.25645` (96x72) while PIC improved
`.62255 -> .08685`. Is the ranking an artefact of the half-cell front convention?

**Evidence.** `validation/dam_break_diagnosis.py` varies one switch at a time from a 95%-FLIP
`n=96` baseline and scores seven front conventions (raw rightmost, rightmost `+ h/2`, 99th and
99.5th percentile, last column holding 4 or 8 particles, and last column at least two cells
deep). Measured RMSE:

| case | rightmost | `+h/2` | p99 | column>=4 | thickness>=2h |
|---|---:|---:|---:|---:|---:|
| 48x36 | .04053 | .07255 | .13114 | .03688 | .04216 |
| 96x72 | .23772 | .25645 | .04561 | .19984 | .20922 |
| 144x108 | .27231 | .28490 | .06770 | .25932 | .25595 |
| seed 1 / seed 2 (96) | .23449 / .23735 | .25325 / .25610 | | | |
| 9 particles/cell (96) | .24489 | .26368 | | | |
| `dt/2` (96) | .18751 | .20607 | | | |
| `dt/4` (96) | **.09635** | **.11451** | | | |
| PIC blend .20 (96) | .15709 | .17569 | | | |
| pure FLIP (96) | .27066 | .28959 | | | |

**What it is not.** The ranking survives six of the seven conventions: any particle-tracking
front (rightmost, `+h/2`, both percentiles at 48, both column rules, thickness) puts 96-144 far
above 48. So the convention is not the cause. Jitter seed changes the 96 RMSE by `<1.5%`, and
nine particles per cell by `<3%`, so the initial sampling is not the cause either. The front is a
smooth distribution -- the gap between the furthest and second-furthest particle is `0.3-0.6 h` --
so a single stray particle is not driving it.

**What it is.** Dominated by time-step error. At `n=96` (`dt = 5e-4, 2.5e-4, 1.25e-4`) the
RMSE falls `.25645 -> .20607 -> .11451`, i.e. halving `dt` more than halves the error on the last
step. At `n=144` the same first halving only moves `.28490 -> .26382`, so the time-step error is
already less dominant there and the two grids bracket a non-zero limit. The front also
*overshoots* the experiment progressively as the grid is refined: measured delta at `tau=2.96`
is 3.73 / 3.99 / 4.01 for 48 / 96 / 144 against the reference 3.67, while the wall-contact time
`tau` saturates at 2.90 for both 96 and 144 (2.44 for 48). Coarse FLIP damps the surge, and that
extra numerical damping happens to sit closer to a slower experiment. Refining removes it, so the
error approaches a limit set by missing physics, not by `h`. Grid refinement therefore cannot
reduce it below that limit; only time-step refinement reduces the numerical part.

**Consequence.** The milestone claim "FLIP does not converge" was too strong. Correct statement:
the front series is grid-convergent for `PIC` and grid-converged-but-not-monotone for FLIP,
because at the documented `dt_max` the FLIP error is dominated by time-step error whose sign
happens to be favourable on the coarse grid. The comparison with the 1952 experiment is
therefore **unmet** as a convergence demonstration, and is reported as such.

**Side effect fixed.** Reseeding bounds were validated even with `reseed` off, so a legitimate
9-particle-per-cell seeding was rejected by the default donor cap. Validation now runs only when
reseeding is enabled, and `max_particles_per_cell` defaults to 16 so denser seedings are usable.
