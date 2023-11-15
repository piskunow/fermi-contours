"""Marching Squares module."""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Union
from warnings import warn

import numpy as np
import numpy.typing as npt


PairInt = tuple[int, int]
PairFloat = tuple[float, float]
LPInt = list[PairInt]
LPFloat = list[PairFloat]


def _dummy(x: float, y: float) -> float:
    return x + y


class MarchingSquares:
    """Marching square class.

    Parameters
    ----------
    grid_values: ndarray of floats with shape `(n, m)`, optional
        Values of the function on a 2-dimensional coordinates grid.
    func: callable
        Function that returns floats for each point in a 2-dimensional
        coordinates space.
        If `grid_values` is provided, `func` is used to resolve saddle
        points.
        If `grid_values` is not provided, then `func`, `bounds` and `res`
        must be provided. The `grid_values` is obtained using a
        2-dimensional coordinates grid.
    bounds : ndarray-like of shape (2, 2)
        Bounds in the x- and y-axis `((x_min, x_max), (y_min, y_max))`,
        optional. If not provided, the bounds are assumed to be
        `((0, n), (0, m))`.
    res: int, optional
        Number of linear points to subdivide each of the axis intervals.
        If bounds is not provided, `res` is not used, and the shape
        of the `grid_values` is used instead.
    open_contours : bool
        Wheather to allow open contours or raise an error when contours
        do not close on themselves.
    periodic: bool, default to 'False'
        If 'True', the 2-dimensional coordinates grid has periodic
        boundaries. Thus, the point `(n, j)` is equivalent to `(0, j)`,
        and `(i, m)` equivalent to `(i, 0)`. If bounds are provided,
        then `x_max` should be mapped to `x_min`, and `y_max` to `y_min`.
    """

    def __init__(
        self,
        grid_values: Optional[npt.NDArray[np.float_]] = None,
        func: Optional[Callable[[float, float], float]] = None,
        bounds: Optional[
            Union[tuple[PairInt, PairInt], tuple[PairFloat, PairFloat]]
        ] = None,
        res: Optional[Union[int, PairInt]] = None,
        open_contours: bool = True,
        periodic: bool = False,
    ) -> None:
        """Initialize MarchingSquares."""
        # compute the coordinates grid
        if bounds is None and grid_values is None:
            raise ValueError("Either 'bounds' or 'grid_values' must be provided.")
        if bounds is None:
            n, m = np.array(grid_values).shape
            bounds = ((0, n), (0, m))
            # discard `res`
            if res is not None:
                raise ValueError("'res' is discarded when 'bounds' is not provided.")
            res = (n, m)
        self.bounds = bounds
        if res is None:
            raise ValueError("'res' is discarded when 'bounds' is not provided.")
        elif isinstance(res, int):
            _res = (res, res)
        elif len(res) == 2:
            _res = res
        else:
            raise ValueError("'res' must be 'None', an 'int', or a pair of ints.")
        self.res = _res

        # the coordinates are computed with a property, it needs to know
        # about the periodic condition (to include the endpoint or not)
        self.periodic = periodic

        # compute the grid values
        if grid_values is None:
            if func is None:
                raise ValueError(
                    "'func' must be a callable when 'grid_values' is 'None'."
                )
            self.grid_values = self._compute_grid_values(func)
        else:
            self.grid_values = np.atleast_2d(grid_values)
        self.func = func
        self.open_contours = open_contours

    def __call__(self, level: int = 0) -> list[LPFloat]:
        """Calcualte the Fermi contours for a Fermi level.

        Sets values for the attributes defined below.

        Parameters
        ----------
        level : float, default to '0'
            Isolevel.

        Returns
        -------
        contour_paths: list of lists of pairs of floats.
            Each list has numerical interpolated points along the path.
        """
        contours_cells, contour_paths = self._find_contours(level)
        # check for repeated cells and return only the largest
        digested_contour_cells: list[Any] = []
        digested_contour_paths: list[Any] = []
        for c_cells, v_cells in zip(contours_cells, contour_paths):
            # check that this does not belong to digest
            updated = False
            for digest, paths in zip(
                digested_contour_cells,
                digested_contour_paths,
            ):
                if not set(c_cells).isdisjoint(digest):
                    digest.update(c_cells)
                    updated = True
                    paths.update(v_cells)
            if not updated:
                digested_contour_cells.append(set(c_cells))
                digested_contour_paths.append(v_cells)

        return [list(d.values()) for d in digested_contour_paths]

    @property
    def grid_points(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Start grid to find the contours."""
        x1, x2 = self.bounds[0]
        y1, y2 = self.bounds[1]
        n_x, n_y = self.res
        endpoint = not self.periodic
        x_array = np.linspace(x1, x2, n_x, endpoint=endpoint, dtype=np.float_)
        y_array = np.linspace(y1, y2, n_y, endpoint=endpoint, dtype=np.float_)
        return x_array, y_array

    def _compute_grid_values(
        self, func: Callable[[float, float], float]
    ) -> npt.NDArray[np.float_]:
        x_array, y_array = self.grid_points
        grid_values: npt.NDArray[np.float_] = np.ndarray(self.res, dtype=np.float_)

        for ix in range(self.res[0]):
            for iy in range(self.res[1]):
                grid_values[ix, iy] = func(x_array[ix], y_array[iy])
        return grid_values

    def _find_contours(
        self, level: float
    ) -> tuple[list[LPInt], list[dict[PairInt, PairFloat]]]:
        """Get the coarse grid coordinates of one contour, or set of contours.

        This is a fast version of the MarchinSquares algorithm, that uses
        the computed values of all cells simultaneously.

        Parameters
        ----------
        level : float
            Isolevel of the contours.
        """
        # start with a particular cell (needed for later, so that the
        # refinement works easily)
        n_x, n_y = self.res
        x_array, y_array = self.grid_points

        # construct binary grid of 0's and 1's regions
        gridvals = self.grid_values < level
        cells = (
            (gridvals[:-1, :-1] << 0)
            + (gridvals[1:, :-1] << 1)
            + (gridvals[1:, 1:] << 2)
            + (gridvals[:-1, 1:] << 3)
        )

        grid_binary = 1 * gridvals
        # save the values that change after a x-shift
        roll_x = np.logical_xor(grid_binary, np.roll(grid_binary, shift=-1, axis=0))
        # save the values that change after a y-shift
        roll_y = np.logical_xor(grid_binary, np.roll(grid_binary, shift=-1, axis=1))
        # find the union of x-shifts and y-shifts
        roll_xy = np.logical_or(roll_x, roll_y)
        # the indices of the region contours are the `xys`
        xys = {
            (x, y)
            for (x, y) in np.argwhere(roll_xy)
            if np.all((x, y) < (n_x - 1, n_y - 1))
        }  # filter last indices

        contours_cells = []
        contour_paths = []

        d_ij: PairInt = (
            0,
            0,
        )  # no direction yet, but needed for marching step reference

        # we go through all the nontrivial grid points, but not visit the same
        # loop twice.

        mod = None
        if self.periodic:
            mod = (self.res[0] - 1, self.res[1] - 1)
            origin = (self.bounds[0][0], self.bounds[1][0])
            periods = (
                self.bounds[0][1] - self.bounds[0][0],
                self.bounds[1][1] - self.bounds[1][0],
            )

        while xys:
            initial_point = min(xys)  # lexicographical minimum of tuples
            single_contour: LPInt = []
            single_path: dict[PairInt, PairFloat] = dict()
            xys.discard(initial_point)
            ij = initial_point

            last_xys = []

            while True:
                # make sure we went through all the indices in the contour
                i, j = ij
                middle_k = (
                    (x_array[i] + x_array[i + 1]) / 2,
                    (y_array[j] + y_array[j + 1]) / 2,
                )

                try:
                    d_ij = marching_step(cells[ij], self.func, middle_k, d_ij)
                except RuntimeError:
                    warn("Saddle point not resolved.")
                    if self.func is None:
                        warn(
                            "Saddle point not resolved because 'func' is not provided."
                        )
                    break

                xy = marching_cell_values(
                    ij, d_ij, self.grid_values, x_array, y_array, level, mod=mod
                )
                i, j = np.array(ij, dtype=int) + d_ij
                if mod is not None:
                    i = (i + mod[0]) % mod[0]
                    j = (j + mod[1]) % mod[1]
                    _x, _y = xy
                    _x = origin[0] + (_x - origin[0] + periods[0]) % periods[0]
                    _y = origin[1] + (_y - origin[1] + periods[1]) % periods[1]
                    xy = (_x, _y)

                ij = (i, j)

                try:
                    xys.remove(ij)
                except KeyError:
                    warn(f"Stepping outside the initial path with cell {ij}.")
                    if ij in last_xys:
                        break
                    last_xys.append(ij)

                # check if the next cell exists in the grid
                i, j = ij
                if (0 <= i < n_x) and (0 <= j < n_y):
                    single_contour.append(ij)
                    single_path[ij] = xy
                else:
                    contours_cells.append(single_contour)
                    contour_paths.append(single_path)
                    if self.open_contours:
                        break
                    else:
                        raise RuntimeError(
                            "Contour goes outisde the bounds. "
                            "Set 'open_contours' to True if "
                            "that is the expected behavior."
                        )

                if ij == initial_point:
                    # The contour closes on itself
                    contours_cells.append(single_contour)
                    contour_paths.append(single_path)
                    break

        return contours_cells, contour_paths


def marching_cell_values(
    ij: PairInt,
    d_ij: PairInt,
    grid_values: npt.NDArray[np.float_],
    x_array: npt.NDArray[np.float_],
    y_array: npt.NDArray[np.float_],
    level: float = 0.0,
    mod: Optional[PairInt] = None,
) -> PairFloat:
    """Return the interpolated values where the contour crosses the new boundary.

    Parameters
    ----------
    ij : pair of ints
        Zeroth position of the cell on the grid.
    d_ij : pair of ints
        Direction where the marching cell is moving.
    grid_values: ndarray of shape `(n, m)`
        Values to use in the linear interpolation.
    x_array, y_array: ndarrays of floats
        Coordinates of the grid points.
    level : float, default to '0'
        The level of the isosurface, that is, contour in 2d.
    mod: paif of ints, optional
        If provided, it indicates the size of the periodic grid.
        Contours will wrap around the boundaries.
    """
    if mod is None:
        _mod = (2**32, 2**32)
    else:
        _mod = mod

    def values(i: int, j: int) -> float:
        return grid_values[i % _mod[0], j % _mod[1]]  # type: ignore

    i, j = ij
    if d_ij[0] == 1:
        # o-----o-----x
        # | old | new |
        # o-----o-----x
        new_values = (
            values(i + 1, j),
            values(i + 1, j + 1),
        )
        x = x_array[i + 1]
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        y = y_array[j] + weight * (y_array[j + 1] - y_array[j])
    elif d_ij[0] == -1:
        # x-----o-----o
        # | new | old |
        # x-----o-----o
        new_values = (
            values(i, j),
            values(i, j + 1),
        )
        x = x_array[i]
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        y = y_array[j] + weight * (y_array[j + 1] - y_array[j])
    elif d_ij[1] == 1:
        # x-----x
        # | new |
        # o-----o
        # | old |
        # o-----o
        new_values = (
            values(i, j + 1),
            values(i + 1, j + 1),
        )
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        x = x_array[i] + weight * (x_array[i + 1] - x_array[i])
        y = y_array[j + 1]
    elif d_ij[1] == -1:
        # o-----o
        # | old |
        # o-----o
        # | new |
        # x-----x
        new_values = (
            values(i + 1, j),
            values(i, j),
        )
        weight = (level - new_values[1]) / (new_values[0] - new_values[1])
        x = x_array[i] + weight * (x_array[i + 1] - x_array[i])
        y = y_array[j]
    else:
        raise ValueError(f"The displacement {d_ij} is trivial.")

    return x, y


def marching_step(
    cell: int,
    func: Optional[Callable[[float, float], float]],
    middle: PairFloat,
    d_ij: PairInt,
) -> PairInt:
    """Return the direction to the next cell.

    The parameters `func`, `middle` and `d_ij` are only accessed
    when they are necessary to resolve saddle-point ambiguities
    (i.e. ``cell == 0b0101`` or ``cell == 0b1010``).
    """
    try:
        return MARCHING_STEPS[cell]
    except KeyError as err:
        if func is None:
            raise RuntimeError(f"cell {str(cell)} shouldn't happen ...") from err
        else:
            if cell == 0b0101:
                if func(*middle) < 0:
                    if d_ij == (0, 1):
                        new_d_ij = (1, 0)
                    elif d_ij == (0, -1):
                        new_d_ij = (-1, 0)
                    else:
                        new_d_ij = (0, 0)
                else:
                    if d_ij == (0, 1):
                        new_d_ij = (-1, 0)
                    elif d_ij == (0, -1):
                        new_d_ij = (1, 0)
                    else:
                        new_d_ij = (0, 0)
            elif cell == 0b1010:
                if func(*middle) < 0:
                    if d_ij == (1, 0):
                        new_d_ij = (0, -1)
                    elif d_ij == (-1, 0):
                        new_d_ij = (0, 1)
                    else:
                        new_d_ij = (0, 0)
                else:
                    if d_ij == (1, 0):
                        new_d_ij = (0, 1)
                    elif d_ij == (-1, 0):
                        new_d_ij = (0, -1)
                    else:
                        new_d_ij = (0, 0)
            else:
                new_d_ij = (0, 0)

            if new_d_ij == (0, 0):
                raise RuntimeError("Inconsistent direction and cell value.") from err
            return new_d_ij


# tables definition
"""MARCHING_STEPS - Assigns a direction to the square corner values."""
MARCHING_STEPS: dict[int, PairInt] = {
    0b0000: (1, 0),
    0b0001: (-1, 0),
    0b0100: (1, 0),
    0b0010: (0, -1),
    0b1000: (0, 1),
    0b0011: (-1, 0),
    0b0110: (0, -1),
    0b1100: (1, 0),
    0b1001: (0, 1),
    0b0111: (-1, 0),
    0b1110: (0, -1),
    0b1101: (1, 0),
    0b1011: (0, 1),
}
