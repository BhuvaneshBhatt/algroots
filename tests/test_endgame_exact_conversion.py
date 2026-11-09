"""Endpoint certification must retain every bit of the numerical center."""

import mpmath as mp
import pytest
import sympy as sp

from algroots.endgames import _mp_rational


@pytest.mark.parametrize("text", ["0", "0.3", "-1.5", "1e-200", "1e200"])
def test_endpoint_center_is_exact_binary_rational(text):
    with mp.workdps(80):
        value = mp.mpf(text)
        sign, mantissa, exponent, _ = value._mpf_
        expected = sp.Rational((-1 if sign else 1) * int(mantissa)) * sp.Rational(2) ** int(
            exponent
        )
        actual = _mp_rational(value)
        assert actual.is_Rational is True
        assert actual == expected


@pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
def test_nonfinite_endpoint_center_is_rejected(text):
    with pytest.raises(ValueError, match="finite"):
        _mp_rational(mp.mpf(text))
