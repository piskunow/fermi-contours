"""Test cases for the MarchingSquares class."""
import numpy as np
from numpy.testing import assert_allclose

from fermi_contours.marching_squares import MarchingSquares


def surface(x: float, y: float) -> float:
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


level_open = 1.1

val_contour_paths_open = [
    [
        (-0.5263157894736843, 0.9004328254847646),
        (-0.6989414121724066, 0.7777777777777777),
        (-0.736842105263158, 0.741871858007592),
        (-0.8843425113710203, 0.5555555555555554),
        (-0.9473684210526316, 0.4361380424746075),
        (-0.9907748538011696, 0.33333333333333326),
        (-1.0376884340480832, 0.11111111111111116),
        (-1.0376884340480832, -0.11111111111111116),
        (-0.9907748538011696, -0.33333333333333337),
        (-0.9473684210526316, -0.4361380424746075),
        (-0.8843425113710202, -0.5555555555555556),
        (-0.736842105263158, -0.741871858007592),
        (-0.6989414121724064, -0.7777777777777778),
        (-0.5263157894736843, -0.9004328254847646),
        (-0.31611842105263155, -1.0),
    ],
    [
        (0.5263157894736841, -0.9004328254847647),
        (0.6989414121724065, -0.7777777777777778),
        (0.7368421052631575, -0.7418718580075925),
        (0.8843425113710202, -0.5555555555555556),
        (0.9473684210526314, -0.4361380424746081),
        (0.9907748538011696, -0.33333333333333337),
        (1.0376884340480832, -0.11111111111111116),
        (1.0376884340480832, 0.11111111111111116),
        (0.9907748538011696, 0.33333333333333326),
        (0.9473684210526314, 0.436138042474608),
        (0.8843425113710204, 0.5555555555555554),
        (0.7368421052631575, 0.7418718580075926),
        (0.6989414121724066, 0.7777777777777777),
        (0.5263157894736841, 0.9004328254847647),
        (0.3161184210526316, 1.0),
    ],
]
val_contours_cells_open = [
    [
        (7, 8),
        (6, 8),
        (6, 7),
        (5, 7),
        (5, 6),
        (4, 6),
        (4, 5),
        (4, 4),
        (4, 3),
        (4, 2),
        (5, 2),
        (5, 1),
        (6, 1),
        (6, 0),
        (7, 0),
    ],
    [
        (11, 0),
        (12, 0),
        (12, 1),
        (13, 1),
        (13, 2),
        (14, 2),
        (14, 3),
        (14, 4),
        (14, 5),
        (14, 6),
        (13, 6),
        (13, 7),
        (12, 7),
        (12, 8),
        (11, 8),
    ],
]


def test_single_contour() -> None:
    """Test closed single contour."""
    squares = MarchingSquares(
        func=surface,
        res=(20, 10),
        bounds=((-2, 2), (-1, 1)),
    )

    contours_cells, contour_paths = squares._find_contours(level=level)

    contour = squares(level=level)[0]

    assert_allclose(contours_cells[0], val_contours_cells)
    assert_allclose(contour_paths[0], contour)
    assert_allclose(val_contour_paths, contour)


def test_open_contours() -> None:
    """Test closed single contour."""
    squares = MarchingSquares(
        func=surface,
        res=(20, 10),
        bounds=((-2, 2), (-1, 1)),
    )

    contours_cells, contour_paths = squares._find_contours(level=level_open)
    pruned_cells, pruned_paths = squares._check_repeated(contours_cells, contour_paths)

    contours = squares(level=level_open)

    assert_allclose(pruned_cells[0], val_contours_cells_open[0])
    assert_allclose(pruned_cells[1], val_contours_cells_open[1])
    assert_allclose(pruned_paths[0], contours[0])
    assert_allclose(pruned_paths[1], contours[1])
    assert_allclose(val_contour_paths_open[0], contours[0])
    assert_allclose(val_contour_paths_open[1], contours[1])
