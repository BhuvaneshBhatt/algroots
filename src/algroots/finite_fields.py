"""Validated finite coefficient fields and univariate FLINT factorization adapters."""

from itertools import product

import sympy as sp
from flint import fmpz_mod_poly_ctx, fq_default_ctx, fq_default_poly_ctx
from sympy.polys.agca.extensions import ExtensionElement, MonogenicFiniteExtension
from sympy.polys.domains.field import Field
from sympy.polys.polyerrors import CoercionFailed


class FiniteFieldExtension(MonogenicFiniteExtension, Field):
    """An exact GF(p^d) coefficient domain with a specified irreducible polynomial.

    The quotient representation interoperates with SymPy polynomial and module
    algorithms. Univariate factorization is delegated to python-flint's finite
    extension fields. Reducible or composite-characteristic moduli are rejected.
    """

    def __init__(self, modulus):
        if (
            not isinstance(modulus, sp.Poly)
            or not modulus.is_univariate
            or not modulus.domain.is_FiniteField
        ):
            raise ValueError("a univariate polynomial over a prime field is required")
        if (
            not sp.isprime(modulus.domain.characteristic())
            or modulus.degree() < 2
            or not modulus.is_irreducible
        ):
            raise ValueError(
                "an irreducible polynomial of degree at least two over GF(p) is required"
            )
        super().__init__(modulus.monic())
        self._flint_context = fq_default_ctx(
            modulus=fmpz_mod_poly_ctx(int(self.characteristic()))(
                [int(c) for c in reversed(self.modulus.all_coeffs())]
            ),
            var=str(self.symbol),
            p=int(self.characteristic()),
        )

    def __eq__(self, other):
        return type(self) is type(other) and self.modulus == other.modulus

    __hash__ = MonogenicFiniteExtension.__hash__

    def of_type(self, value):
        return isinstance(value, ExtensionElement) and value.ext == self

    def convert(self, value, base=None):
        if isinstance(value, ExtensionElement):
            if value.ext == self:
                return value
            raise CoercionFailed(
                "finite extension fields require an explicit coefficient embedding"
            )
        return super().convert(value, base)

    def exquo(self, left, right):
        return left / right

    @property
    def order(self):
        """The finite field cardinality p^degree."""
        return int(self.characteristic()) ** self.rank

    def _flint(self, value):
        value = self.convert(value)
        return self._flint_context([int(c) for c in reversed(value.rep.to_list())])

    def _from_flint(self, value):
        return self.convert(sum(int(c) * self.symbol**i for i, c in enumerate(value.to_list())))


def finite_field(characteristic, degree=1, *, name="a", modulus=None):
    """Construct GF(p) or GF(p^degree), using an explicit irreducible modulus if supplied."""
    if (
        isinstance(characteristic, bool)
        or not isinstance(characteristic, (int, sp.Integer))
        or not sp.isprime(characteristic)
    ):
        raise ValueError("the finite-field characteristic must be prime")
    if isinstance(degree, bool) or not isinstance(degree, (int, sp.Integer)) or degree < 1:
        raise ValueError("degree must be a positive integer")
    p = int(characteristic)
    variable = sp.Symbol(name) if isinstance(name, str) else name
    if not isinstance(variable, sp.Symbol):
        raise ValueError("the extension generator must be a Symbol or a name")
    if modulus is not None:
        polynomial = (
            modulus if isinstance(modulus, sp.Poly) else sp.Poly(modulus, variable, modulus=p)
        )
        if polynomial.domain != sp.GF(p):
            raise ValueError("the modulus must have the requested characteristic")
        if degree != 1 and polynomial.degree() != degree:
            raise ValueError("the modulus degree disagrees with degree")
        return FiniteFieldExtension(polynomial)
    if degree == 1:
        return sp.GF(p)
    context = fq_default_ctx(p, int(degree), str(variable))
    coefficients = context.modulus().coeffs()
    polynomial = sp.Poly(
        sum(int(c) * variable**i for i, c in enumerate(coefficients)), variable, modulus=p
    )
    return FiniteFieldExtension(polynomial)


def is_finite_field(domain):
    """Whether a domain is a validated prime or finite-extension coefficient field."""
    return (domain.is_FiniteField and bool(sp.isprime(domain.characteristic()))) or isinstance(
        domain, FiniteFieldExtension
    )


def field_order(domain):
    """Return the order of a supported finite field, rejecting other coefficient domains."""
    if not is_finite_field(domain):
        raise TypeError("a supported finite coefficient field is required")
    return (
        domain.order if isinstance(domain, FiniteFieldExtension) else int(domain.characteristic())
    )


def field_elements(domain):
    """Iterate native elements of a finite coefficient field without duplicates."""
    if isinstance(domain, FiniteFieldExtension):
        p = int(domain.characteristic())
        for coefficients in product(range(p), repeat=domain.rank):
            yield domain.convert(sum(c * domain.symbol**i for i, c in enumerate(coefficients)))
    elif is_finite_field(domain):
        for i in range(int(domain.characteristic())):
            yield domain.convert(i)
    else:
        raise TypeError("a supported finite coefficient field is required")


def factor_univariate(polynomial):
    """Factor an exact univariate polynomial over a supported finite coefficient field."""
    if not isinstance(polynomial, sp.Poly) or not polynomial.is_univariate:
        raise ValueError("a univariate Poly is required")
    field = polynomial.domain
    if not is_finite_field(field):
        raise TypeError("a supported finite coefficient field is required")
    if not isinstance(field, FiniteFieldExtension):
        return polynomial.factor_list()
    coefficients = [field._flint(c) for c in reversed(polynomial.rep.to_list())]
    native = fq_default_poly_ctx(field._flint_context)(coefficients)
    unit, factors = native.factor()
    result = []
    for factor, exponent in factors:
        data = {(i,): field._from_flint(c) for i, c in enumerate(factor.coeffs())}
        result.append((sp.Poly.from_dict(data, *polynomial.gens, domain=field), int(exponent)))
    return field.to_sympy(field._from_flint(unit)), result


__all__ = [
    "FiniteFieldExtension",
    "finite_field",
    "is_finite_field",
    "field_order",
    "field_elements",
    "factor_univariate",
]
