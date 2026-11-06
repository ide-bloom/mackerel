"""The 2D incompressible solver: one place where advection, forces, viscosity and the
pressure projection are put together into a time step.

The equations
-------------
We solve the incompressible Navier-Stokes equations with density ``rho = 1``::

    du/dt + (u.grad) u = -grad p + nu laplacian u + f
    div u = 0

on a MAC grid.  Time advances with the classic four-stage *projection* (Chorin/temam) split,
one stage per physical term:

1. **Advect** ``u* = A(u^n)`` with semi-Lagrangian / MacCormack / BFECC.  Taking advection
   first (rather than after the projection) means the velocity that gets advected is already
   divergence free, which is the usual choice for graphics-style solvers.
2. **Forces** ``u** = u* + dt f`` -- buoyancy, gravity, user forces; all explicit.
3. **Diffuse** ``(I - dt nu L) u*** = u**`` -- implicit, unconditionally stable (see
   :mod:`mackerel.diffuse`).  With ``nu = 0`` this stage is skipped exactly, not approximately.
4. **Project** ``u^{n+1} = u*** - dt grad p`` with ``laplacian p = div u*** / dt``, solved by
   any of the four solvers from :mod:`mackerel.solvers`.

Why the split is allowed
------------------------
Splitting the operators introduces an error of order ``dt`` per step in general (a commutator
of advection and projection).  For the smooth, low-Reynolds-number flows we validate against it
is small, and it is measured, not assumed: the Taylor-Green decay rate check and the
Poiseuille profile check in ``validation/`` both compare against closed-form answers.

Stability
---------
Advection is stable for any ``dt`` but *accurate* only for CFL ``= |u| dt / h <~ 1``.
Diffusion is unconditionally stable.  The only hard requirement is the CFL number, so the
solver can pick ``dt`` itself (``cfl`` parameter) and reports the largest CFL it actually ran.

Boundary conditions in the time step
------------------------------------
Every stage is followed by :func:`mackerel.grid.apply_velocity_bc`, because each stage can
damage a prescribed normal velocity: interpolation near the wall, a force with a normal
component, and the pressure gradient all write to boundary faces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mackerel.advect import AdvectionOptions, Scheme, advect_scalar, advect_velocity
from mackerel.diagnostics import Diagnostics, diagnose, kinetic_energy, max_cfl
from mackerel.diffuse import diffuse, diffuse_scalar
from mackerel.grid import Array, BoundaryConditions, Grid, apply_velocity_bc
from mackerel.obstacles import apply_solid_velocity
from mackerel.project import PoissonSystem, ProjectionResult, project_velocity

__all__ = [
    "BUOYANCY_GRAVITY",
    "Simulation",
    "SimulationConfig",
    "StepInfo",
    "buoyancy_force",
    "run",
]


#: Standard gravity used by the Rayleigh-Taylor / plume demos, in grid units (``L = 1``).
BUOYANCY_GRAVITY = 9.81


@dataclass
class SimulationConfig:
    """Everything that controls a run.

    ``dt`` is either given directly or derived each step from ``cfl`` (``dt = cfl h / |u|max``).
    ``nu`` is the kinematic viscosity; ``0`` gives the inviscid Euler equations, in which case
    the diffusion stage is skipped entirely.
    """

    dt: float | None = None
    cfl: float = 1.0
    nu: float = 0.0
    scheme: Scheme = Scheme.SEMI_LAGRANGIAN
    limiter: bool = True
    order: int = 2
    solver: str = "conjugate_gradient"
    pressure_tol: float = 1e-9
    pressure_maxiter: int = 500
    scalar_diffusivity: float = 0.0
    buoyancy: tuple[float, float] = (0.0, 0.0)
    gravity: float = 0.0
    scalar_buoyancy: float = 0.0
    advect_scalars: bool = True

    def __post_init__(self) -> None:
        # Accept the CLI's string spelling as well as the enum, without silently treating an
        # unrecognised string as one of the corrected schemes.
        raw_scheme: object = self.scheme
        if isinstance(raw_scheme, str):
            self.scheme = Scheme(raw_scheme)

    def advection_options(self) -> AdvectionOptions:
        return AdvectionOptions(scheme=self.scheme, limiter=self.limiter, order=self.order)


@dataclass(frozen=True)
class StepInfo:
    """What happened during one time step."""

    time: float
    dt: float
    cfl: float
    pressure_iterations: int
    pressure_residual: float
    div_max: float
    div_l2: float


@dataclass
class Simulation:
    """A 2D incompressible flow on a MAC grid.

    State: ``u``, ``v`` (MAC faces), ``scalars`` (name -> cell-centred field) and the elapsed
    ``time``.  Every call to :meth:`step` returns a :class:`StepInfo` and leaves the state
    divergence free to the pressure tolerance.
    """

    grid: Grid
    bc: BoundaryConditions
    config: SimulationConfig = field(default_factory=SimulationConfig)
    u: Array = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    v: Array = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    scalars: dict[str, Array] = field(default_factory=dict)
    solid: np.ndarray | None = None
    solid_velocity: tuple[float, float] = (0.0, 0.0)
    time: float = 0.0

    def __post_init__(self) -> None:
        self.bc.validate()
        if self.u.size == 0:
            self.u = self.grid.zeros_u()
        if self.v.size == 0:
            self.v = self.grid.zeros_v()
        self.u = np.asarray(self.u, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)
        apply_velocity_bc(self.u, self.v, self.bc)
        apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)
        self._system: PoissonSystem | None = None
        self.history: list[Diagnostics] = []

    # -- construction helpers -----------------------------------------------------------
    @classmethod
    def taylor_green(
        cls,
        grid: Grid,
        nu: float = 0.0,
        bc: BoundaryConditions | None = None,
        **config: object,
    ) -> Simulation:
        """Start from the exact Taylor-Green velocity field (periodic by default)."""
        from mackerel.analytic import taylor_green as tg
        from mackerel.grid import periodic_bc

        u, v = tg(grid)
        return cls(
            grid=grid,
            bc=bc if bc is not None else periodic_bc(),
            config=SimulationConfig(nu=nu, **config),  # type: ignore[arg-type]
            u=u,
            v=v,
        )

    @classmethod
    def at_rest(
        cls,
        grid: Grid,
        bc: BoundaryConditions,
        **config: object,
    ) -> Simulation:
        """Quiescent fluid, for the lid-driven cavity and other wall-driven problems."""
        return cls(grid=grid, bc=bc, config=SimulationConfig(**config))  # type: ignore[arg-type]

    # -- forcing ------------------------------------------------------------------------
    def add_scalar(self, name: str, field: Array | float = 0.0) -> Array:
        """Register or overwrite a cell-centred scalar (dye, temperature)."""
        if isinstance(field, float):
            value = np.full(self.grid.shape_cell, field, dtype=np.float64)
        else:
            value = np.array(field, dtype=np.float64, copy=True)
        self.scalars[name] = value
        return value

    def apply_force(
        self,
        fx: Array | float = 0.0,
        fy: Array | float = 0.0,
        dt: float = 1.0,
    ) -> None:
        """Add a cell-centred body acceleration ``(fx, fy)`` to the velocity (explicit).

        A cell force is mapped to the faces that carry the velocity component by averaging,
        which is the discrete adjoint of the interpolation used by the diagnostics.
        """
        if isinstance(fx, float):
            self.u += dt * fx
        else:
            self.u[:, 1:-1] += dt * 0.5 * (np.asarray(fx)[:, :-1] + np.asarray(fx)[:, 1:])
        if isinstance(fy, float):
            self.v += dt * fy
        else:
            self.v[1:-1, :] += dt * 0.5 * (np.asarray(fy)[:-1, :] + np.asarray(fy)[1:, :])
        apply_velocity_bc(self.u, self.v, self.bc)

    def _buoyancy(self, dt: float) -> None:
        """``f = g beta T`` downwards, plus any constant body force from the config."""
        cfg = self.config
        fx, fy = cfg.buoyancy
        if fx != 0.0 or fy != 0.0:
            self.apply_force(fx, fy, dt)
        if cfg.scalar_buoyancy != 0.0 and self.scalars:
            # the first scalar registered is the one that drives buoyancy
            name = next(iter(self.scalars))
            field = self.scalars[name]
            weight = float(np.mean(field))
            # The scalar is temperature-like: larger values are lighter and rise in the
            # positive-y direction when ``gravity`` is positive.
            f = cfg.gravity * cfg.scalar_buoyancy * (field - weight)
            self.apply_force(0.0, f, dt)

    # -- time step ----------------------------------------------------------------------
    def _system_for(self) -> PoissonSystem:
        if self._system is None:
            solid = (
                np.zeros(self.grid.shape_cell, dtype=bool)
                if self.solid is None
                else np.asarray(self.solid, dtype=bool)
            )
            self._system = PoissonSystem.from_bc(
                self.grid.ny, self.grid.nx, self.grid.h, self.bc, solid=solid
            )
        return self._system

    def set_solid(
        self, solid: np.ndarray | None, velocity: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """Replace the immersed obstacle and invalidate its cached pressure system."""
        self.solid = None if solid is None else np.asarray(solid, dtype=bool).copy()
        self.solid_velocity = velocity
        self._system = None
        apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)

    def dt_for_cfl(self) -> float:
        """Time step that gives the configured CFL, or the fixed ``dt`` if one is set."""
        cfg = self.config
        if cfg.dt is not None:
            return float(cfg.dt)
        speed = self.max_speed()
        # With a nearly quiescent flow the raw CFL estimate can be enormous (and a body
        # force would then be applied over an unphysically long first step).  A cell-crossing
        # time is a conservative, deterministic upper bound; users needing a stricter accuracy
        # limit can set ``dt`` explicitly.
        if speed <= 1e-14:
            return float(self.grid.h)
        return float(min(self.grid.h, cfg.cfl * self.grid.h / speed))

    def max_speed(self) -> float:
        uc = np.abs(self.grid.u_to_centers(self.u))
        vc = np.abs(self.grid.v_to_centers(self.v))
        return float(np.max(uc + vc))

    def step(self, dt: float | None = None) -> StepInfo:
        """Advance one time step and return what happened."""
        cfg = self.config
        step_dt = self.dt_for_cfl() if dt is None else float(dt)
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")
        cfl = max_cfl(self.u, self.v, self.grid, step_dt)

        # 1. advection of velocity and of every scalar
        self.u, self.v = advect_velocity(
            self.u, self.v, self.grid, self.bc, step_dt, cfg.advection_options()
        )
        apply_velocity_bc(self.u, self.v, self.bc)
        apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)
        if cfg.advect_scalars and self.scalars:
            for name, field in self.scalars.items():
                self.scalars[name] = advect_scalar(
                    field, self.u, self.v, self.grid, self.bc, step_dt,
                    cfg.advection_options(),
                )

        # 2. forces
        self._buoyancy(step_dt)

        # 3. viscosity (skipped exactly when inviscid)
        if cfg.nu > 0.0:
            res = diffuse(
                self.u, self.v, self.grid, self.bc, cfg.nu, step_dt,
                tol=1e-10, maxiter=400,
            )
            self.u, self.v = res.u, res.v
            apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)
        # Scalar diffusivity is independent of kinematic viscosity (e.g. dye in an inviscid
        # liquid), so it must not be nested inside the velocity-diffusion branch.
        if cfg.scalar_diffusivity > 0.0 and self.scalars:
            for name, field in self.scalars.items():
                self.scalars[name] = diffuse_scalar(
                    field, self.grid, self.bc, cfg.scalar_diffusivity, step_dt
                )

        # 4. projection
        apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)
        proj: ProjectionResult = project_velocity(
            self.u,
            self.v,
            self.grid,
            self.bc,
            step_dt,
            system=self._system_for(),
            solver=cfg.solver,
            tol=cfg.pressure_tol,
            maxiter=cfg.pressure_maxiter,
            solid=self.solid,
        )
        self.u, self.v = proj.u, proj.v
        self.time += step_dt
        apply_velocity_bc(self.u, self.v, self.bc)
        apply_solid_velocity(self.u, self.v, self.solid, self.solid_velocity)

        info = StepInfo(
            time=self.time,
            dt=step_dt,
            cfl=cfl,
            pressure_iterations=proj.iterations,
            pressure_residual=proj.residual,
            div_max=proj.div_max,
            div_l2=proj.div_l2,
        )
        self.history.append(diagnose(self.u, self.v, self.grid, step_dt, self.bc))
        return info

    # -- convenience --------------------------------------------------------------------
    def run_steps(
        self,
        steps: int,
        callback: Callable[[Simulation, StepInfo], None] | None = None,
    ) -> list[StepInfo]:
        """Run ``steps`` steps, calling ``callback(self, info)`` after each one."""
        infos: list[StepInfo] = []
        for _ in range(steps):
            info = self.step()
            infos.append(info)
            if callback is not None:
                callback(self, info)
        return infos

    def run_until(
        self,
        end_time: float,
        callback: Callable[[Simulation, StepInfo], None] | None = None,
        max_steps: int = 1_000_000,
    ) -> list[StepInfo]:
        """Run until ``self.time >= end_time``."""
        infos: list[StepInfo] = []
        while self.time < end_time - 1e-12 and len(infos) < max_steps:
            # Do not step past the requested final time; this also makes validation runs
            # comparable when the CFL-derived dt changes from step to step.
            remaining = end_time - self.time
            info = self.step(dt=min(self.dt_for_cfl(), remaining))
            infos.append(info)
            if callback is not None:
                callback(self, info)
        return infos

    def energy(self) -> float:
        return kinetic_energy(self.u, self.v, self.grid)


def buoyancy_force(
    grid: Grid, field: Array, gravity: float = BUOYANCY_GRAVITY, beta: float = 1.0
) -> Array:
    """Cell-centred vertical acceleration ``g beta (T - mean T)`` from a scalar field.

    The scalar is temperature-like, so hot fluid rises.  Subtracting the mean keeps the total
    weight out of the problem: only horizontal differences in ``T`` should drive motion,
    otherwise a uniform scalar accelerates the whole fluid and the boundaries have to absorb
    an unbalanced momentum flux.
    """
    return gravity * beta * (np.asarray(field, dtype=np.float64) - float(np.mean(field)))


def run(
    grid: Grid,
    bc: BoundaryConditions,
    config: SimulationConfig,
    end_time: float,
    callback: Callable[[Simulation, StepInfo], None] | None = None,
    u0: Array | None = None,
    v0: Array | None = None,
) -> Simulation:
    """Create a simulation from ``(u0, v0)`` and integrate it to ``end_time``."""
    sim = Simulation(
        grid=grid,
        bc=bc,
        config=config,
        u=np.empty((0, 0), dtype=np.float64) if u0 is None else u0,
        v=np.empty((0, 0), dtype=np.float64) if v0 is None else v0,
    )
    sim.run_until(end_time, callback)
    return sim
