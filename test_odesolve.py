#
# Test functions for odesolve
#
# The tests here check that the functions defined in odesolve.py produce the
# right outputs. To run the tests make sure that odesolve.py and this file
# test_odesolve.py are in the same directory and then run this file.
#

import numpy as np
from odesolve import euler, rk4, solveto, odesolve


def test_euler():
    def f(x, t):
        return x + t

    assert euler(f, 0, 0, 1) == 0
    assert euler(f, 1, 0, 1) == 2
    assert euler(f, 1, 0, 2) == 3
    assert euler(f, 1, 0, 0.5) == 1.5
    assert euler(f, 0, 1, 1) == 1
    assert euler(f, 1, 1, 1) == 3
    assert euler(f, 1, 1, 2) == 5
    assert euler(f, 1, 1, 0.5) == 2.0


def test_solveto():
    def f(x, t):
        return x

    assert solveto(f, 0, 0, 1, 1) == 0
    assert solveto(f, 1, 0, 1, 1) == 2
    assert solveto(f, 1, 0, 1, 0.5) == 2.25
    assert solveto(f, 1, 0, 1, 0.25) == 2.44140625
    assert solveto(f, 1, 0, 1, 0.3) == 2.4167

    xtrue = 2.71828182845905
    for h in [0.1, 0.01, 0.001, 0.003]:
        xguess = solveto(f, 1, 0, 1, h)
        assert abs(xguess - xtrue) < 2*h

    def f(x, t):
        return x + t

    xtrue = -11
    for h in [0.1, 0.01, 0.001, 0.003, 1e-5]:
        xguess = solveto(f, -2, 1, 10, h)
        assert abs(xguess - xtrue) < 1e-10 / h**0.5


def isclose(x1, x2):
    return abs(x1 - x2) / (abs(x1) + abs(x2)) < 1e-10


def test_rk4():
    def f(x, t):
        return x

    assert rk4(f, 0, 0, 1) == 0
    assert isclose(rk4(f, 1, 0, 1), 2.708333333333333)
    assert isclose(rk4(f, 1, 1, 3), 16.375)


def test_rk4_solveto():
    def f(x, t):
        return x
    xtrue = 2.71828182845905
    for h in [0.1, 0.01, 0.001, 0.003]:
        xguess = solveto(f, 1, 0, 1, h, rk4)
        assert abs(xguess - xtrue) < h**4


def test_arrays():
    def f(X, t):
        x1, x2 = X
        dx1dt = x2
        dx2dt = -x1
        dXdt = [dx1dt, dx2dt]
        return np.array(dXdt)

    X0 = np.array([1, -1])
    assert np.allclose(euler(f, X0, 0, 1), np.array([0, -2]))
    assert np.allclose(rk4(f, X0, 0, 1), np.array([-0.29166667, -1.375]))
    assert np.allclose(solveto(f, X0, 0, 1, 0.25), np.array([-0.30859375, -1.56640625]))
    assert np.allclose(solveto(f, X0, 0, 1, 0.25, rk4), np.array([-0.30112267, -1.38177358]))


def test_odesolve():

    def f(x, t):
        return x

    x0 = [1]
    tvals = np.linspace(0, 1, 5)
    expected_euler = np.array([[1. , 1.28386503, 1.64830942, 2.11620682, 2.71692393]])
    expected_rk4   = np.array([[1. , 1.28402542, 1.64872127, 2.11700002, 2.71828183]])
    guess_euler = odesolve(f, x0, tvals, 0.001, euler)
    guess_rk4   = odesolve(f, x0, tvals, 0.001, rk4)
    assert np.allclose(expected_euler, guess_euler)
    assert np.allclose(expected_rk4, guess_rk4)


test_functions = [
    test_euler,
    test_rk4,
    test_solveto,
    test_rk4_solveto,
    test_arrays,
    test_odesolve,
]


passed = 0
for func in test_functions:
    print('Checking', func.__name__, '...')
    exc = None
    try:
        func()
    except Exception as e:
        print('FAILED')
        failed = True
        exc = e
    else:
        print('PASSED')
        passed += 1
        failed = False

    if failed:
        break

print(passed, 'tests passed out of', len(test_functions), 'tests')

if failed:
    print(func.__name__, 'failed with exception:')
    raise exc from None
