"""§17.1 -- DP5 observed convergence order on linear and nonlinear problems."""
import numpy as np

from dp5_reference import integrate, tableau


def _order(f, y0, exact, t_end, dts):
    errs = []
    for dt in dts:
        n = int(round(t_end / dt))
        y = integrate(f, y0, dt, n)
        errs.append(abs(float(y[0]) - exact))
    errs = np.asarray(errs)
    return np.log2(errs[:-1] / errs[1:])


def test_tableau_rows_consistent():
    T = tableau()
    # every DP5 row must sum to its stage abscissa
    assert np.isclose(T["A21"], 1 / 5)
    assert np.isclose(T["A31"] + T["A32"], 3 / 10)
    assert np.isclose(T["A41"] + T["A42"] + T["A43"], 4 / 5)
    assert np.isclose(T["A51"] + T["A52"] + T["A53"] + T["A54"], 8 / 9)
    assert np.isclose(T["A61"] + T["A62"] + T["A63"] + T["A64"] + T["A65"], 1.0)
    assert np.isclose(T["B1"] + T["B3"] + T["B4"] + T["B5"] + T["B6"], 1.0)
    # b - b* must sum to zero (both rows are consistent)
    assert abs(T["E1"] + T["E3"] + T["E4"] + T["E5"] + T["E6"] + T["E7"]) < 1e-14


def test_order_linear():
    # y' = -y, y(0) = 1
    orders = _order(lambda y: -y, np.array([1.0]), np.exp(-1.0), 1.0,
                    [0.1, 0.05, 0.025, 0.0125])
    assert abs(orders.mean() - 5.0) < 0.2, orders


def test_order_nonlinear():
    # logistic y' = y (1 - y), y(0) = 1/2  ->  y = e^t / (1 + e^t)
    import math
    te = 10.0
    exact = math.exp(te) / (1.0 + math.exp(te))
    orders = _order(lambda y: y * (1.0 - y), np.array([0.5]), exact, te,
                    [te / 40, te / 80, te / 160, te / 320])
    assert abs(orders.mean() - 5.0) < 0.2, orders


def test_error_estimate_is_fourth_order():
    from dp5_reference import dp5_step
    T = tableau()
    e = []
    for h in (0.1, 0.05, 0.025):
        _, err = dp5_step(lambda y: -y * y, np.array([1.0]), h, T)
        e.append(abs(float(err[0])))
    e = np.asarray(e)
    orders = np.log2(e[:-1] / e[1:])
    assert np.all(orders > 4.5) and np.all(orders < 6.5), orders
