"""Cross-check the evidence claims made in the user guides."""

from dataclasses import replace

import pytest
import sympy as sp

from algroots import polysolve
from algroots.errors import RootCertificationError

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "equations,multiplicity,geometric",
    [
        ((x**2, y**2 - 1), 4, 2),
        ((x - y**2, y**2 - 2), 2, 2),
        ((x - sp.sqrt(2) * y, y**2 - 1), 2, 2),
    ],
)
def test_documented_presolve_order_and_recombination_contract(equations, multiplicity, geometric):
    original = polysolve(equations, (x, y), method="rur", recognize=False, presolve=False)
    recombined = (2 * equations[0] + 3 * equations[1], -equations[0] - equations[1])
    transformed = polysolve(recombined, (y, x), method="rur", recognize=False)
    assert original.total_multiplicity == transformed.total_multiplicity == multiplicity
    assert len(original.roots) == len(transformed.roots) == geometric
    for root in original.roots:
        assert min(
            max(abs(sp.N(a - b, 30)) for a, b in zip(root, other[::-1], strict=True))
            for other in transformed.roots
        ) < sp.Rational(1, 10**25)


def test_documented_endpoint_proof_does_not_promote_global_status():
    result = polysolve((x**2,), (x,), recognize=False, certify="required")
    assert result.completeness.status == "conditional"
    assert result.root_certifications[0].status == "certified"
    certificate = result.root_certifications[0].certificate
    assert certificate.multiplicity == 2 and certificate.verify()
    assert not replace(certificate, multiplicity=1).verify()


def test_documented_auto_and_required_refusals():
    result = polysolve(
        (x**2 - 2,), (x,), recognize=False, certify="auto", certification_max_dimension=1
    )
    assert len(result.roots) == 2
    assert all(
        a.status == "unavailable" and a.certificate is None and a.error
        for a in result.root_certifications
    )
    with pytest.raises(RootCertificationError):
        polysolve(
            (x**2 - 2,), (x,), recognize=False, certify="required", certification_max_dimension=1
        )
