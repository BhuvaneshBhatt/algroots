"""Discover a quadratic orbit from one supplied seed.

Guarantee: Numerical orbit discovery; no automatic exact count.
Run from the repository root: python examples/14_seeded_monodromy.py
"""

import sympy as sp

from algroots.continuation import PathTrackerOptions
from algroots.monodromy import discover_monodromy_orbit


def main():
    x = sp.Symbol("x")
    orbit = discover_monodromy_orbit(
        (x**2 - 1,),
        (x,),
        ((1,),),
        max_loops=4,
        radius=2,
        random_seed=0,
        recognize=False,
        options=PathTrackerOptions(
            initial_step=0.03, max_step=0.07, initial_digits=40, residual_digits=24
        ),
    )
    assert len(orbit.roots) == 2 and orbit.all_paths_successful
    print("Discovered roots:", orbit.roots)
    print("Loops:", orbit.loops_completed, "Paths:", orbit.paths_tracked)
    print("Seeds nad numerical stopping do not give an exact completeness certificate.")


if __name__ == "__main__":
    main()
