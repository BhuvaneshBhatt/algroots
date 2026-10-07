"""Return repeated roots while retaining aligned distinct-root evidence."""

import sympy as sp

from algroots import polysolve


def main():
    x = sp.Symbol("x")
    result = polysolve(
        [(x + 2) * (x - 1) ** 3],
        [x],
        recognize=False,
        multiplicity="required",
        root_mode="with_multiplicity",
        root_order="required",
    )
    assert result.multiplicities == (1, 3)
    assert len(result.roots) == 2
    assert len(result) == 4
    assert tuple(result.iter_roots()) == result.output_roots
    assert result.output_roots == (result.roots[0],) + (result.roots[1],) * 3
    assert result.ordering.status == "certified"
    print("Distinct points:", result.roots)
    print("Certified multiplicities:", result.multiplicities)
    print("Repeated view:", result.output_roots)
    print("Ordering evidence:", result.ordering)


if __name__ == "__main__":
    main()
