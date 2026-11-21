"""§17.7 -- triangular matrix formatting and content."""
import numpy as np

from izhiclust.stats import render_triangular, triangular_frame, COEFFS


def test_triangular_frame_contents():
    rng = np.random.default_rng(0)
    P = np.abs(rng.normal(size=(200, 4)) @ np.array([[1, .5, 0, 0], [0, 1, .3, 0],
                                                     [0, 0, 1, .2], [0, 0, 0, 1.0]])) + 0.05
    df = triangular_frame(P)
    R = np.corrcoef(P, rowvar=False)
    sd = np.std(P, axis=0, ddof=1)
    M = df[list(COEFFS)].to_numpy()
    for i in range(4):
        assert np.isclose(M[i, i], sd[i])
        for j in range(i):
            assert np.isclose(M[i, j], R[i, j])
        for j in range(i + 1, 4):
            assert np.isnan(M[i, j])
    assert np.isclose(df["sd_log_a"].iloc[0], np.std(np.log10(P[:, 0]), ddof=1))
    assert np.isnan(df["sd_log_a"].to_numpy()[1:]).all()


def test_upper_triangle_renders_empty():
    rng = np.random.default_rng(1)
    P = np.abs(rng.normal(size=(50, 4))) + 0.1
    text = render_triangular(triangular_frame(P))
    lines = [l for l in text.splitlines() if l.strip()]
    header, body = lines[0], lines[1:]
    assert header.split()[:4] == list(COEFFS)
    # row i must have exactly i + 1 numeric entries among the 4 coefficient
    # columns (lower triangle + diagonal), the rest blank
    for i, line in enumerate(body[:4]):
        cells = line.split()
        assert cells[0] == COEFFS[i]
        n_num = sum(1 for c in cells[1:] if _isnum(c))
        # lower triangle + diagonal, plus the sd_log_a column on the first row
        expect = i + 1 + (1 if i == 0 else 0)
        assert n_num == expect, (i, line)


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
