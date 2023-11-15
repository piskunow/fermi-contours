"""Test cases for the MarchingSquares class."""
import numpy as np
from numpy.testing import assert_allclose

from fermi_contours.marching_squares import MarchingSquares


def surface(x, y):
    """Construct a simple surface."""
    return x**2 + y**2


level = 0.05

val_contour_paths = np.array(
    [
        (-0.16837638076673164, -0.11111111111111116),
        (-0.10526315789473695, -0.17090258541089565),
        (0.10526315789473673, -0.17090258541089573),
        (0.16837638076673156, -0.11111111111111116),
        (0.16837638076673156, 0.11111111111111116),
        (0.10526315789473673, 0.17090258541089576),
        (-0.10526315789473695, 0.17090258541089565),
        (-0.16837638076673164, 0.11111111111111116),
        (-0.16837638076673164, -0.11111111111111116),
    ]
)
val_contours_cells = [
    (8, 4),
    (8, 3),
    (9, 3),
    (10, 3),
    (10, 4),
    (10, 5),
    (9, 5),
    (8, 5),
    (8, 4),
]
val_contour_paths_dict = zip(val_contours_cells, val_contour_paths)


def test_single_contour() -> None:
    """Test closed single contour."""
    squares = MarchingSquares(
        func=surface,
        res=[20, 10],
        bounds=[[-2, 2], [-1, 1]],
    )

    contours_cells, contour_paths = squares._find_contours(level=level)

    contour = squares(level=level)[0]

    assert_allclose(contours_cells[0], val_contours_cells)
    assert_allclose(list(contour_paths[0].values()), contour)
    assert_allclose(val_contour_paths, contour)
