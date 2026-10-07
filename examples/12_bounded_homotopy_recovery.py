"""Recover and certify a singular finite endpoint.

Guarantee: Exact finite-root accounting; no path proof.
Run from the repository root: python examples/12_bounded_homotopy_recovery.py
"""

import sympy as sp

from algroots import HomotopyRecoveryOptions, polysolve


def main():
    x = sp.Symbol("x")
    options = HomotopyRecoveryOptions(max_paths=4, gamma_attempts=1, max_retry_rounds=1)
    result = polysolve(
        (x**2,),
        (x,),
        method="homotopy",
        homotopy_recovery=True,
        recovery_options=options,
        digits=30,
        recognize=False,
    )
    assert result.completeness.status == "certified" and len(result.roots) == 1
    assert result.root_certifications[0].certificate.multiplicity == 2
    print("Evidence:", result.completeness)
    for record in result.recovery_records:
        print(
            "Path, retry, precision, proof:",
            record.path_index,
            record.retry_round,
            record.working_digits,
            record.certification_status,
        )


if __name__ == "__main__":
    main()
