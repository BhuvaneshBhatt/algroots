"""Regressions for ideal preservation, budgets, ownership and cancellation."""

import concurrent.futures
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import sympy as sp
from sympy.polys.rings import ring

from algroots.native_f4 import compiled_available as native_available

if native_available():
    from algroots import _native_f4
from algroots.ff_normal_form import compiled_available, normal_form
from algroots.groebner import compute_groebner_basis


@pytest.mark.parametrize("backend", ["native"])
@pytest.mark.skipif(not native_available(), reason="compiled kernel is not installed")
def test_certified_native_iterator_uses_same_generators(backend):
    R, x, y = ring("x,y", sp.GF(101), order="grevlex")
    assert compute_groebner_basis(iter([x + y]), R, backend=backend) == (x + y,)


@pytest.mark.parametrize("backend", ["python", "auto", "compiled"])
def test_division_limits_include_steps_and_quotient(backend):
    if backend == "compiled" and not compiled_available():
        pytest.skip("optional sparse extension unavailable")
    for steps, terms in [(1, 100), (100, 1)]:
        with pytest.raises(OverflowError, match="resource limit"):
            normal_form(
                {(10,): 1},
                [{(1,): 1, (0,): -1}],
                101,
                backend=backend,
                max_steps=steps,
                max_terms=terms,
            )


@pytest.mark.skipif(not native_available(), reason="compiled kernel is not installed")
def test_native_input_does_not_swallow_keyboard_interrupt():
    class InterruptedExponent:
        def __index__(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _native_f4.compute([{(InterruptedExponent(),): 1}], 101, 1)


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.skipif(not native_available(), reason="compiled kernel is not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process signals")
def test_native_signal_cancels_before_finishing(reentrant):
    library = Path(_native_f4.__file__).resolve()
    code = r"""
import importlib.util,json,time,signal
spec=importlib.util.spec_from_file_location('_native_f4',LIBRARY)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
if REENTRANT:
 def handler(signum,frame):
  # Input parsing in a nested call must recognize that the outer checkpoint
  # already holds the GIL, including its own monomial-interning checkpoints.
  native.compute([{(i,):1 for i in range(5000)},{(0,):1}],101,1,'lex',max_terms=10000)
  raise KeyboardInterrupt
 signal.signal(signal.SIGINT,handler)
n=7
fs=[{tuple(int((j-i)%n<d) for j in range(n)):1 for i in range(n)} for d in range(1,n)]
fs.append({(1,)*n:1,(0,)*n:-1})
print('ready',flush=True)
t=time.monotonic()
try:
 native.compute(fs,1009,n,'grevlex',max_pairs=100000,max_basis=2048,max_terms=300000,max_reduction_steps=20000000)
 print('finished',flush=True)
except KeyboardInterrupt:
 print(json.dumps({'cancelled':True,'elapsed':time.monotonic()-t}),flush=True)
""".replace("LIBRARY", repr(str(library))).replace("REENTRANT", repr(reentrant))
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        time.sleep(0.1)
        sent = time.monotonic()
        child.send_signal(signal.SIGINT)
        out, err = child.communicate(timeout=10)
        assert child.returncode == 0, err
        assert json.loads(out)["cancelled"]
        assert time.monotonic() - sent < 1.5
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


@pytest.mark.skipif(not native_available(), reason="compiled kernel is not installed")
def test_parallel_native_calls_keep_computation_state_separate():
    def compute(i):
        p = [7, 101][i % 2]
        order = ["lex", "grevlex"][i % 2]
        fs = [{(2, 0): 1, (0, 1): -1}, {(1, 1): 1, (0, 0): -1}]
        return _native_f4.compute(fs, p, 2, order)[0]

    expected = [compute(i) for i in range(12)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(compute, range(12))) == expected
