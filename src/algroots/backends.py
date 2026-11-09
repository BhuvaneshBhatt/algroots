"""Discover Gröbner engines and their coefficient, order, and installation requirements."""

from dataclasses import dataclass

import sympy as sp

from .groebner_errors import GroebnerBackendUnavailable
from .native_f4 import NativeF4Limits

_ORDERS = ("lex", "grlex", "grevlex")
_FIELDS = (
    "QQ",
    "prime fields",
    "finite extension fields",
    "number fields",
    "rational function fields",
    "algebraic function fields and towers",
)


def _extension_available():
    try:
        from . import _native_extension
    except ImportError:
        return False
    return hasattr(_native_extension, "compute")


@dataclass(frozen=True)
class GroebnerBackendInfo:
    """One engine's current availability and supported mathematical inputs.

    ``limits_type`` names its operation-budget class, or is None when no such
    class is accepted. Availability describes dependencies, not completion on
    a particular ideal. All accelerated public results still require verification.
    """

    name: str
    available: bool
    domains: tuple[str, ...]
    orders: tuple[str, ...]
    limits_type: type | None
    installation: str
    reason: str | None = None
    max_characteristic: int | None = None
    max_variables: int | None = None


def _backend_info(name):
    native = name == "native"
    if native:
        from .native_f4 import compiled_available

        available = compiled_available()
        installation = (
            "Install a native algroots wheel, or build from source with "
            "ALGROOTS_BUILD_NATIVE_F4=1 python -m pip install ."
        )
        return GroebnerBackendInfo(
            name,
            available,
            _FIELDS[:2]
            + (("finite extension fields",) if _extension_available() else ())
            + (
                ("algebraic function fields and towers",)
                if name == "native" and _extension_available()
                else ()
            )
            + (("number fields",) if name == "native" and _extension_available() else ())
            + (
                ("rational function fields over number fields",)
                if name == "native" and _extension_available()
                else ()
            )
            + (("rational function fields over QQ",) if name == "native" else ()),
            _ORDERS,
            NativeF4Limits,
            installation,
            None if available else "the compiled F4 extension could not be imported",
            1000000007,
            64,
        )
    if name == "sympy":
        return GroebnerBackendInfo(
            name, True, _FIELDS, _ORDERS, None, "python -m pip install sympy"
        )
    raise ValueError("backend must be sympy or native")


def groebner_backends():
    """Describe native tiled F4 and portable SymPy availability and domains."""
    return {name: _backend_info(name) for name in ("sympy", "native")}


def _require_backend(name, domain, nvars):
    from .finite_fields import FiniteFieldExtension
    from .groebner import UnsupportedGroebnerDomain

    info = _backend_info(name)
    rational = domain.is_QQ
    prime = domain.is_FiniteField
    finite_extension = isinstance(domain, FiniteFieldExtension)
    if name == "native" and finite_extension and not _extension_available():
        raise GroebnerBackendUnavailable(
            "native finite extension fields require the FLINT native extension; "
            "build with ALGROOTS_BUILD_NATIVE_EXTENSION=1 or use backend='sympy'"
        )
    parameter = (
        domain.is_FractionField
        and domain.domain == sp.QQ
        and "rational function fields over QQ" in info.domains
    )
    mixed = (
        domain.is_FractionField
        and domain.domain.is_AlgebraicField
        and "rational function fields over number fields" in info.domains
    )
    ground = domain.domain if domain.is_FractionField else domain
    if name == "native" and ground.is_AlgebraicField and not _extension_available():
        raise GroebnerBackendUnavailable(
            f"backend {name!r} for domain {domain} requires the FLINT native extension; install FLINT development files and build with ALGROOTS_BUILD_NATIVE_EXTENSION=1, or use backend='sympy'"
        )
    tower = getattr(domain, "is_AlgebraicFunctionField", False)
    if tower and name == "native" and not _extension_available():
        raise GroebnerBackendUnavailable(
            "native algebraic function fields require the FLINT native extension; use backend='sympy' or build with ALGROOTS_BUILD_NATIVE_EXTENSION=1"
        )
    supported = (
        (tower and name == "native")
        or name == "sympy"
        or rational
        or parameter
        or mixed
        or (domain.is_AlgebraicField and "number fields" in info.domains)
        or (prime and "prime fields" in info.domains)
        or (finite_extension and "finite extension fields" in info.domains)
    )
    if not supported:
        raise UnsupportedGroebnerDomain(
            f"backend {name!r} does not support domain {domain}; supported: "
            f"{', '.join(info.domains)}. Use backend='sympy' for this exact field."
        )
    if (
        (prime or finite_extension)
        and info.max_characteristic
        and domain.characteristic() > info.max_characteristic
    ):
        raise UnsupportedGroebnerDomain(
            f"backend {name!r} supports prime moduli at most {info.max_characteristic}; "
            f"got {domain}. Use backend='sympy' to preserve this field."
        )
    if info.max_variables and nvars > info.max_variables:
        raise UnsupportedGroebnerDomain(
            f"backend {name!r} supports at most {info.max_variables} variables; got {nvars}. "
            "Use backend='sympy'."
        )
    if not info.available:
        raise GroebnerBackendUnavailable(
            f"backend {name!r} for domain {domain} is unavailable: {info.reason}. "
            f"{info.installation} Or select backend='sympy'."
        )
