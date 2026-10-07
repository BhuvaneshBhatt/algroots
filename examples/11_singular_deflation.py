"""Deflate and replay an exact singular-root proof.

Guarantee: Local exact deflation; finite ideal required.
Run from the repository root: python examples/11_singular_deflation.py
"""

import sympy as sp

from algroots.deflation import deflate_isolated_root


def main():
    x = sp.Symbol("x")
    result = deflate_isolated_root((x**3,), (x,), (0,), max_stages=4)
    assert result.certificate.multiplicity == 3 and result.regular
    assert result.verify()
    print("Original multiplicity:", result.certificate.multiplicity)
    print("Deflation stages:", result.stages)
    print("Stopping reason:", result.stopping_reason)


if __name__ == "__main__":
    main()
