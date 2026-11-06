# The mathematics of mackerel, in plain language

This document explains every step the solver takes. It is written to be *read aloud*: each
section states the idea in words first, then the equation, then what the code does with it.
Every number quoted here was produced by a command in this repository; the command is given.

Contents

1. [What we are solving](#1-what-we-are-solving)
2. [Continuity: nothing is created or destroyed](#2-continuity-nothing-is-created-or-destroyed)
3. [Why the grid is staggered](#3-why-the-grid-is-staggered)
4. [Splitting the equations: advect, diffuse, push, project](#4-splitting-the-equations)
5. [Advection](#5-advection)
6. [Diffusion](#6-diffusion)
7. [Pressure projection](#7-pressure-projection)
8. [Boundary conditions](#8-boundary-conditions)
9. [Diagnostics: what the numbers mean](#9-diagnostics)
10. [Particles and free surfaces (PIC/FLIP)](#10-particles-and-free-surfaces)

---

## 1. What we are solving

We solve the incompressible Navier–Stokes equations for a fluid of constant density
(`rho = 1`, so we never write it):

```
du/dt + (u · ∇)u = -∇p + ν ∇²u + f          (momentum)
∇ · u = 0                                    (incompressibility)
```

Read in order: the fluid accelerates (`du/dt`) because it is *carried along by itself*
(`(u·∇)u`), because of pressure differences (`-∇p`), because of friction (`ν ∇²u`), and
because of outside pushes such as gravity (`f`). The second line says the fluid is
incompressible: nowhere can fluid pile up or be created.

`u = (u, v)` is the velocity in two dimensions (`x` and `y`), `p` is pressure divided by
density, `ν` is the kinematic viscosity and `f` any body force.

Two auxiliary quantities are used constantly:

* **Vorticity** `ω = ∂v/∂x - ∂u/∂y`. It measures local spin, positive counter-clockwise.
  In 2-D it is a single scalar per point, and because of incompressibility it obeys the
  same advection–diffusion equation as dye, plus a *stretching* term that vanishes in 2-D.
  That is why 2-D vorticity is such a good diagnostic.
* **Enstrophy** `Z = ∫ ½ω² dA`. Total squared spin. For smooth 2-D flow it only ever
  decreases (viscosity) or stays constant (ideal flow, at the resolution of the continuous
  equations). A code that *grows* enstrophy in an inviscid test is creating spurious small
  scales — a classic numerical artefact.

---

## 2. Continuity: nothing is created or destroyed

`∇·u = 0` in words: take a tiny box; the amount of fluid entering through its left and
bottom faces equals the amount leaving through its right and top faces.

On the grid, cells have size `h`, `u` values sit on the vertical faces and `v` values on
the horizontal faces, so the box's faces are exactly where the velocities are stored. That
makes the discrete version of the statement *exact*, not an approximation:

```
div[i,j] = (u[j, i+1] - u[j, i]) / h + (v[j+1, i] - v[j, i]) / h
```

(that is `mackerel.grid.divergence`). "Divergence-free to machine precision" in this
project always means: this array is smaller than about `1e-12` in the max norm.

---

## 3. Why the grid is staggered

This is the single most important structural decision in the whole code, so here is the
argument from first principles.

**The failure mode.** A natural first attempt puts *all* quantities — both velocity
components and pressure — at the same points (cell centres). Then the pressure gradient has
to be a centred difference, `(p[i+1] - p[i-1]) / 2h`. Now consider the pressure field

```
p[i, j] = (-1)^(i+j)
```

which alternates `+1, -1, +1, -1, ...` like a checkerboard. Its centred difference is
`((-1) - (-1)) / 2h = 0` *everywhere*. The gradient operator has a non-trivial null space,
and the checkerboard mode is in it: it produces no force at all, so the projection step
can never remove it, and it grows freely from round-off until it dominates the solution.
The pressure field looks like a chessboard and the velocity field is unaffected — a
"checkerboard pressure" artefact. It is not a bug you can tune away: the operator is
genuinely singular on that mode.

**The fix.** Put each velocity component on the faces it flows through: `u` on vertical
faces, `v` on horizontal faces, scalars (pressure, dye, temperature) at cell centres. This
is the MAC (Marker-And-Cell) staggered grid, from Harlow and Welch (1965).

**Why it works.** The pressure gradient between two neighbouring cells is now
`(p[i] - p[i-1]) / h` — a *nearest-neighbour* difference evaluated exactly on the face where
the velocity lives, and the velocity that the projection adjusts is on that same face. There
is no interpolation anywhere in the pressure/velocity coupling: the divergence at a cell is
built from exactly the four faces that the gradient on those faces touches. The composite
operator `div(grad p)` is the plain 5-point Laplacian, which is non-singular (up to the
constant mode, which is handled explicitly — see §7). The checkerboard mode is now the
*most strongly* felt mode instead of the invisible one.

**The numerical demonstration.** `mackerel validate foundations` prints:

```
checkerboard (collocated vs staggered)
  h                        0.0625
  collocated max |grad p|  0.000e+00
  staggered  max |grad p|  32.000000 (= 2/h = 32.000000)
```

On a 16×16 grid with `h = 1/16`, the collocated centred gradient of the checkerboard mode is
*exactly* zero, while the staggered face gradient has magnitude `2/h = 32` — the largest
value any unit-amplitude mode can produce. Figure: `outputs/m1/checkerboard.png`. The same
fact is enforced by `tests/test_grid.py::test_staggered_grid_kills_checkerboard_null_space`.

---

## 4. Splitting the equations

We do not solve the equations as one monolithic system. We use *operator splitting*: in each
time step, apply each physical effect in turn, each with the best method for it:

```
1. add forces            u ← u + dt · f
2. advect                u ← advect(u, dt)          semi-Lagrangian
3. diffuse               u ← solve (I - dt·ν∇²) u = u_old   implicit
4. project               u ← u - dt·∇p   with  ∇²p = ∇·u / dt
```

Steps 1–3 are explicit in the coupling (each uses the previous stage's velocity); step 3 is
implicit *within itself* because explicit viscous diffusion is limited by `dt < h²/(4ν)`,
which is brutal on fine grids. Step 4 is what enforces incompressibility.

Splitting introduces an error of order `dt` per step in the coupling. That is acceptable
here: the dominant accuracy limits are the advection scheme and the grid resolution, and
the split is chosen so each sub-step is individually second-order accurate in space.

---

## 5. Advection

The term `(u·∇)u` says "the velocity is carried by the velocity field". The
semi-Lagrangian method turns that into geometry instead of a derivative:

> To find the velocity at a grid point now, ask *where the fluid that arrives here came
> from* one time step ago, go there, and copy the velocity that was there.

In formulas, with the departure point `x_d = x - dt · u(x)`:

```
u_new(x) = u_old(x_d)
```

The departure point is found by integrating the velocity field backwards in time with
Runge–Kutta 2 (midpoint rule):

```
x_mid = x - (dt/2) u(x)             (interpolate u at x)
x_d   = x - dt · u(x_mid)
```

Then `u_old(x_d)` is evaluated by bilinear interpolation. Details that matter:

* **Interpolating on the staggered grid.** `u` lives at `x = i h`, `y = (j+1/2) h`, so its
  bilinear stencil has a half-cell offset in `y` only; `v` the other way around. The
  offsets are encapsulated in `Grid.u_coords()` / `Grid.v_coords()` and in the sampling
  functions `sample_u` / `sample_v`.
* **Walls.** A backtrace that leaves the domain must not silently keep its old value. We
  sample from *padded* arrays (`grid.pad_velocity`) that contain one ghost layer outside
  every wall, and the ghost values are the mirror of the interior about the wall speed
  (`ghost = 2·wall - interior`). For a tangential component that makes the interpolated
  velocity *at* the wall equal the wall velocity (zero for no-slip, the lid speed for a
  moving lid); for the normal component it is the odd mirror that makes the interpolated
  normal velocity vanish at the wall. Periodic sides wrap instead of mirroring.
* **Numerical diffusion.** Bilinear interpolation is an average, and averaging smooths.
  A sharp blob advected around in a circle slowly turns into a wider, fainter blob. That is
  numerical diffusion — not physical viscosity. Measured in Milestone 2
  (`mackerel validate advection`) and reduced with MacCormack/BFECC, which advect forward
  and backward once and use the difference as an error estimate, then clamp the correction
  so no new maxima or minima appear (a limiter). Every sweep of a staggered component must
  retain that component's offsets. In the M4 BFECC investigation, using the `u` offsets for
  the final `v` sweep caused a measured `-22.9431%` full-solver energy error; restoring
  `(-h/2,-h)` reduced it to `-0.8560%` under the same 64², `ν=.01`, `dt=.0025`, `t=.2` run.

---

## 6. Diffusion

Viscosity makes velocity differences blur out. We solve it implicitly:

```
(I - dt ν ∇²) u_new = u_old
```

`∇²` is the 5-point Laplacian. Implicit means solving a linear system every step; the
system is symmetric positive definite, so conjugate gradient (and multigrid) work well.
The reward is unconditional stability: `dt` is limited by advection, not by `ν/h²`.

---

## 7. Pressure projection

This is the heart of an incompressible solver.

**The idea.** After advection and diffusion, the velocity field `u*` generally has
`∇·u* ≠ 0`: some cells gained fluid, some lost it. Pressure is the Lagrange multiplier that
removes exactly this violation. Take the momentum equation without pressure, then add a
pressure gradient chosen so that the final velocity is divergence-free:

```
u_new = u* - dt ∇p,       with   ∇ · u_new = 0
```

Substituting the first into the second gives the **pressure Poisson equation**:

```
∇² p = (1/dt) ∇ · u*
```

**Why it is called "projection".** Write the equation as `u_new = P u*` where
`P = I - dt ∇(∇²)^(-1) ∇·`. The operator `P` is a projection: it keeps the divergence-free
part of `u*` and throws away the rest. Helmholtz decomposition says any field is a
divergence-free part plus a gradient part; advection and diffusion create the gradient part,
and `P` deletes it.

**Discretisation.** On the MAC grid, `∇·` acts from faces to centres and `∇` from centres
to faces, so `∇² = ∇·∇` becomes the standard 5-point Laplacian
`(p[i+1] - 2p[i] + p[i-1] + p[j+1] - 2p[j] + p[j-1]) / h²`. Because of the staggering, no
interpolation is involved and the discrete operators are exact adjoints of each other.

**Boundary conditions for pressure.**

* At a solid wall the velocity normal to the wall is prescribed. The pressure gradient
  normal to the wall must not change it, which means `∂p/∂n = 0` — a **Neumann**
  condition. Implemented as a ghost pressure equal to the interior neighbour.
* At an open (outflow) boundary the pressure is fixed to a constant, normally zero —
  a **Dirichlet** condition. This is what makes the system non-singular.
* At a periodic side the pressure wraps.

**The null space.** If *every* side is a wall or periodic, only pressure *differences*
matter: adding a constant to `p` changes nothing. The matrix is singular and iterative
solvers drift. We handle it explicitly by projecting out the mean of the pressure correction
each iteration (or by fixing one cell), and we say so in the code rather than hoping.

**Accuracy target.** After projection, the max divergence must be below a tolerance that
each scene states and each test enforces.

---

## 8. Boundary conditions

* **No-slip wall:** fluid sticks to the wall, `u_tangential = wall speed` (zero, or the lid
  speed for the cavity), `u_normal = wall normal speed` (zero for a fixed wall). On a
  staggered grid the normal component lives *on* the wall face and is set directly; the
  tangential component lives half a cell away and is imposed through the ghost value.
* **Free-slip (slip) wall:** `u_normal = 0`, `∂u_tangential/∂n = 0`. Ghost = interior value.
* **Inflow:** the normal velocity is prescribed and points into the domain; scalars take a
  prescribed value.
* **Outflow:** zero-gradient (`∂/∂n = 0`) for velocity and scalars, and pressure Dirichlet.
  This lets flow leave without reflecting, at the cost of not knowing the outflow profile.
* **Periodic:** the two opposite faces are identified, so a particle leaving at `x = lx`
  re-enters at `x = 0`. Faces are stored twice with the first and last kept equal.

---

## 9. Diagnostics

* **Divergence** — see §2. Max and L2 norm, in physical units (`L2 = sqrt(Σ div² h²)`).
* **Kinetic energy** `E = Σ_cells ½(u_c² + v_c²) h²` with `u_c`, `v_c` the face values
  averaged to cell centres. For the Taylor–Green vortex this equals `A²/4 · exp(-4k²νt)`
  exactly (continuum), which is what we validate against.
* **Vorticity** `ω = ∂v/∂x - ∂u/∂y` at cell corners (shape `(ny+1, nx+1)`).
* **Enstrophy** `Z = Σ ½ω² h²`.
* **CFL number** `C = dt · max(|u| + |v|) / h`. For the semi-Lagrangian solver this is a
  *quality* parameter, not a stability limit — the scheme is stable at any `C` — but the
  accuracy of the backtrace degrades as `C` grows past ~1, so we keep it near or below 1.
* **Total scalar** `Σ dye · h²`, which must stay constant under pure advection with
  periodic or closed boundaries.

---

## 10. Particles and free surfaces

Filled in with Milestone 7.

---

## 11. The linear solvers (Milestone 3)

The pressure equation is a linear system `A p = b` with `A = -∇²`. On a MAC grid `A` is
symmetric and positive semi-definite, and its four cheapest familiar solvers are all
implemented here so their costs can be compared instead of assumed.

* **Jacobi.** `p_new = p + ω D⁻¹ (b - A p)`, with `D` the diagonal of `A`. Every cell is
  updated from the previous iterate, so the update is trivially parallel. It removes
  high-frequency error fast and low-frequency error almost not at all: on an `n x n` grid the
  number of sweeps needed grows like `n²`.
* **Gauss-Seidel (red-black).** Two half-sweeps over a checkerboard colouring; each colour is
  updated using the newest values of the other. Vectorised (each colour is a masked array
  assignment, not a loop) and roughly twice as effective per sweep as Jacobi, because
  information travels one cell per colour. Still `O(n²)` sweeps.
* **Conjugate gradient.** Minimises the `A`-norm of the error over a Krylov subspace. Needs
  only `A`-vector products, so it is *matrix-free*: `A` is applied by the same stencil
  function the smoother uses. Convergence in `O(sqrt(κ))` iterations with `κ = O(n²)` the
  condition number, i.e. `O(n)` iterations, with a small constant: measured 122 → 1712
  iterations for `n = 32 → 512`.
* **Multigrid (V-cycle, sparse Galerkin).** Smooth, restrict the residual to a coarser grid,
  recurse, prolong the correction back, smooth again. The low-frequency error that stumps
  the smoothers becomes high-frequency on the coarse grid and is removed there. The coarse
  operators are built as `A_c = R A P` with `R` full-weighting and `P` bilinear prolongation,
  so every level discretises the *same* operator. Measured: **8–9 cycles for every grid from
  32² to 512²**, i.e. a grid-independent convergence factor of ≈0.08 per cycle, and the wall
  time grows with the number of unknowns, not with its square.

`A` is singular when every boundary is a wall or periodic: the constant vector is in its null
space. Three things make that harmless rather than fatal:

1. the right-hand side `div(u*)/dt` has its mean removed (the compatibility condition for a
   solvable pure-Neumann problem),
2. every solver projects the constant mode out of its iterates, so it converges to the
   minimum-norm solution,
3. the caller removes the mean of the returned pressure, so results do not depend on the
   solver's arbitrary constant.

---

## 12. Viscosity and the complete time step (Milestone 4)

The momentum equation is

`∂u/∂t + (u·∇)u = -∇p + ν∇²u + f`, with `∇·u = 0`.

The solver advances one term at a time (a projection/operator-splitting step):

1. **Advect** the velocity and any cell-centred scalars along the current velocity field.
2. **Add forces** explicitly: `u ← u + dt f`. A constant cell-centred force is averaged to
   the MAC faces. A temperature-like scalar produces `f_y = g β (T - mean(T))`, so warm
   fluid rises; subtracting the mean removes a spatially uniform acceleration.
3. **Diffuse implicitly**. Viscosity is a smoothing term. An explicit update would require
   `dt < h²/(4ν)`, which becomes prohibitively small on a fine grid. Backward Euler instead
   solves `(I - dt ν L) u_new = u_old`, where `L` is the five-point Laplacian and
   `α = ν dt/h²`. The matrix is symmetric positive definite after prescribed wall values are
   eliminated, so a diagonal-preconditioned conjugate-gradient iteration solves it without
   assembling a sparse matrix. The same operation diffuses temperature or dye, with its own
   diffusivity.
4. **Project** the result onto the divergence-free velocity space as described in §7.

At a no-slip wall, the tangential velocity is half a cell inside the fluid. Its ghost value is
`u_ghost = 2u_wall - u_inside`. Substituting this into the Laplacian adds `α` to the diagonal
and puts `2αu_wall` on the right-hand side. At a free-slip or outflow wall the normal
gradient is zero, so the outside neighbour is absent. On a periodic seam the duplicated MAC
face is temporarily removed before solving and copied back afterwards; otherwise the linear
system would contain the same unknown twice.

Splitting is not exact: advection and projection do not commute, so there is a first-order-in-
time error in addition to interpolation error. It is therefore important to report both the
analytic Taylor–Green decay and the measured numerical diffusion, rather than claiming that
unconditional stability implies unconditional accuracy. The CFL condition is an accuracy
 guideline for advection; the implicit viscous solve has no explicit diffusive time-step limit.

For the body-force channel, the steady continuous solution is plane Poiseuille flow:
`u(y) = f y(H-y)/(2ν)`, `v = 0`. Comparing cell-centred velocities with this parabola gives a
clean spatial convergence test independent of the Taylor–Green temporal splitting error.

---

## 13. Lid-driven cavity (Milestone 5)

The cavity is a unit square with no-slip walls. The top wall has `u=1`, `v=0`; the other
three walls have `u=v=0`. Reynolds number is `Re=U_lid L/ν`, so with `U_lid=L=1` we set
`ν=1/Re`. The moving lid injects vorticity into the fluid and a large recirculating vortex
forms. At steady state the time derivative is zero, but advection, pressure and viscosity
still balance one another.

The comparison uses the two centreline profiles tabulated by Ghia, Ghia and Shin (1982):
`u(0.5,y)` and `v(x,0.5)`. MAC velocities are first averaged to cell centres, then linearly
interpolated to the reference stations; the known wall velocities are inserted at the two
endpoints. This avoids confusing a half-cell stagger with a physical error. The numerical
run is stopped when the largest velocity change over 100 steps is below `2e-8`, or at the
stated maximum time. Both max and RMS deviations are reported.

One printed Re=400 reference value, `v(x=0.9063)=-0.23827`, is an isolated known tabulation
outlier. It remains in the data file and in the all-point metric; the primary metric excludes
that single cited point so that a reference typo is not mistaken for a solver instability.

---

## 14. Obstacles, buoyancy and vortex shedding (Milestone 6)

An immersed solid is marked at cell centres. A face touching a solid cell is a wall face:
its velocity is prescribed, and the pressure equation has no coupling across it. For a fixed
obstacle that velocity is zero; for a translating obstacle it is the obstacle velocity. This
is a staircase immersed-boundary approximation, so the represented circle changes slightly
with grid resolution. The pressure projection still enforces `div(u)=0` in every fluid cell.

At an open outflow, pressure instead satisfies `p=0` outside. The normal pressure gradient at
that face is therefore `(p_inside-0)/h` (with the appropriate outward sign), not zero. This
distinction matters: using a wall gradient at an outflow leaves a stripe of divergence in the
last cell column.

For a hot scalar anomaly, buoyancy is the vertical acceleration `f_y=gβ(T-mean(T))`. Warm
fluid rises, and subtracting the mean removes the spatially uniform part of the force. A
continually replenished heat source is not a closed conserved-scalar experiment: its integral
is expected to change as heat is added and leaves the top.

A cylinder wake becomes unsteady above the laminar shedding threshold. If a downstream probe
records transverse velocity with dominant frequency `f`, the Strouhal number is `St=fD/U`.
I measure peak-to-peak periods and an FFT independently after discarding the initial
transient. Confinement, staircase geometry and numerical diffusion can shift `St`, so the
measurement is accompanied by the domain and reference relation rather than presented as a
universal constant.

---

## 15. PIC/FLIP and atmospheric free surfaces (Milestone 7)

A particle liquid represents velocity at points `x_p`, while pressure and incompressibility
remain on the MAC grid. Bilinear particle-to-face transfer is a weighted average,
`u_f = sum_p w_pf u_p / sum_p w_pf`; the same weights interpolate grid values back to
particles. The weights are affine and sum to one, including at a domain edge: when an index
is clamped, its fractional coordinate is clamped too. Otherwise a particle near a wall can
receive a negative weight and a nonphysical velocity overshoot.

After gravity and before particle advection, the liquid pressure solves
`-laplacian(p) = -div(u*)/dt`. For a liquid–liquid face the usual neighbour coupling is
retained. For a liquid–air face, the air pressure is atmospheric (`p=0` outside), so the
liquid row gains a Dirichlet diagonal contribution. This is not the solid-wall condition:
at a solid face the normal pressure gradient is zero and the coupling disappears. The
returned pressure is exactly zero in air, and only faces touching liquid are corrected.

The PIC/FLIP blend uses `alpha` as the PIC fraction:

`u_p_new = alpha*u_grid_new + (1-alpha)*(u_p_old + u_grid_new-u_grid_old)`.

PIC directly resamples the grid and is more diffusive; FLIP carries most of the particle
velocity and adds the grid increment, reducing diffusion but retaining more particle noise.
Particle positions are advanced with the updated particle velocity, and a CFL-based step
limits the distance travelled relative to `h`. Closed-wall collisions preserve particles by
clamping position and removing outward normal velocity. Consequently the nominal volume
`N_particles*h^2/particles_per_cell` is exactly conserved in these demos, although this is not
a conservative level-set or VOF interface algorithm.

The dam-break comparison uses `tau=t*sqrt(2g/a)` and `delta=x_front/a`, where `a` is the
initial column width. The front is measured from the actual rightmost particle with a stated
half-cell containing-cell correction. No time shift or parameter fit is applied to the
reference series.
