"""
Copyright (C) (2022) - Pablo M Perez-Piskunow

Code adapted from

Perez-Piskunow, Pablo M, Bovenzi, Nicandro, Akhmerov, Anton R, & Breitkreiz, Maxim. (2021).
Code and data associated with the paper
"Chiral Anomaly Trapped in Weyl Metals: Nonequilibrium Valley Polarization at Zero Magnetic Field"
(1.0.1). Zenodo.
https://doi.org/10.5281/zenodo.4704325

and extended to take into account periodic boundaries and other edge-cases.
"""

from warnings import warn
import numpy as np


class MarchingSquares():
    """Marching square class

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

    def __init__(self, grid_values=None, func=None, bounds=None, res=None,
                 open_contours=True, periodic=False):

        # compute the coordinates grid
        if bounds is None:
            try:
                assert grid_values is not None
            except AssertionError:
                raise ValueError("Either 'bounds' or 'grid_values' must be "
                                 "provided.")
            n, m = np.array(grid_values).shape
            bounds = ((0, n), (0, m))
            # discard `res`
            if res is not None:
                raise ValueError("'res' is discarded when 'bounds' is not "
                                 "provided.")
            res = (n, m)
        self.bounds = bounds
        if isinstance(res, int):
            res = (res, res)
        self.res = res

        # the coordinates are computed with a property, it needs to know
        # about the periodic condition (to include the endpoint or not)
        self.periodic = periodic

        # compute the grid values
        self.func = func
        if grid_values is None:
            if not callable(self.func):
                raise ValueError("'func' must be a callable when "
                                 "'grid_values' is 'None'.")
            self.grid_values = self._compute_grid_values()
        else:
            self.grid_values = np.atleast_2d(grid_values)

        self.open_contours = open_contours

    def __call__(self, level=0):
        """Calcualte the Fermi contours for a Fermi level.

        Sets values for the attributes defined below.

        Parameters
        ----------
        level : float, default to '0'
            Isolevel.

        Returns
        -------
        contour_paths: list of lists of contours.
            Each list has numerical interpolated points along the path.
        """
        contours_cells, contour_paths = self._find_contours(level)
        # check for repeated cells and return only the largest
        digested_contour_cells = []
        digested_contour_paths = []
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
    def grid_points(self):
        x1, x2 = self.bounds[0]
        y1, y2 = self.bounds[1]
        n_x, n_y = self.res
        endpoint = not self.periodic
        x_array = np.linspace(x1, x2, n_x, endpoint=endpoint)
        y_array = np.linspace(y1, y2, n_y, endpoint=endpoint)
        return x_array, y_array

    def _compute_grid_values(self):
        x_array, y_array = self.grid_points
        grid_values = np.ndarray(self.res, dtype=float)

        for ix in range(self.res[0]):
            for iy in range(self.res[1]):
                grid_values[ix, iy] = self.func(x_array[ix], y_array[iy])
        return grid_values

    def _find_contours(self, level):
        """Get the coarse grid coordinates of one contour, or set of contours

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
            (gridvals[:-1, :-1] << 0) + (gridvals[1:, :-1] << 1)
            + (gridvals[1:, 1:] << 2) + (gridvals[:-1, 1:] << 3))

        grid_binary = 1 * gridvals
        # save the values that change after a x-shift
        roll_x = np.logical_xor(grid_binary,
                                np.roll(grid_binary, shift=-1, axis=0))
        # save the values that change after a y-shift
        roll_y = np.logical_xor(grid_binary,
                                np.roll(grid_binary, shift=-1, axis=1))
        # find the union of x-shifts and y-shifts
        roll_xy = np.logical_or(roll_x, roll_y)
        # the indices of the region contours are the `xys`
        xys = {tuple(xy) for xy in np.argwhere(roll_xy)
               if np.all(xy < (n_x - 1, n_y - 1))}  # filter last indices

        contours_cells = []
        contour_paths = []

        d_ij = None  # no direction yet, but needed for marching step reference

        # we go through all the nontrivial grid points, but not visit the same
        # loop twice.

        mod = None
        if self.periodic:
            mod = (self.res[0] - 1, self.res[1] - 1)
            origin = (self.bounds[0][0], self.bounds[1][0])
            periods = (self.bounds[0][1] - self.bounds[0][0],
                       self.bounds[1][1] - self.bounds[1][0])

        while xys:
            initial_point = min(xys)  # lexicographical minimum of tuples
            single_contour = []
            single_path = dict()
            xys.discard(initial_point)
            ij = initial_point

            last_xys = []

            while True:
                # make sure we went through all the indices in the contour
                i, j = ij
                middle_k = ((x_array[i] + x_array[i+1]) / 2,
                            (y_array[j] + y_array[j+1]) / 2)

                try:
                    d_ij = marching_step(cells[ij], self.func, middle_k, d_ij)
                except RuntimeError:
                    warn('Saddle point not resolved.')
                    break
                except TypeError as e:
                    if self.func is None:
                        warn("Saddle point not resolved because 'func' is not provided.")
                        break
                    else:
                        raise TypeError(e)

                xy = marching_cell_values(ij, d_ij, self.grid_values,
                                          x_array, y_array, level, mod=mod)
                i, j = np.array(ij, dtype=int) + d_ij
                if mod is not None:
                    i = (i + mod[0]) % mod[0]
                    j = (j + mod[1]) % mod[1]
                    _x, _y = xy
                    _x = origin[0] + (_x - origin[0] + periods[0]) % periods[0]
                    _y = origin[1] + (_y - origin[1] + periods[1]) % periods[1]
                    xy = tuple([_x, _y])

                ij = tuple([i, j])

                try:
                    xys.remove(ij)
                except KeyError:
                    warn(f"Stepping outside the initial path with cell {ij}.")
                    if ij in last_xys:
                        break
                    last_xys.append(ij)

                # check if the next cell exists in the grid
                try:
                    i, j = ij
                    assert i >= 0 and j >= 0
                    _ = cells[ij]
                    single_contour.append(ij)
                    single_path[ij] = xy
                except (IndexError, AssertionError):
                    contours_cells.append(single_contour)
                    contour_paths.append(single_path)
                    if self.open_contours:
                        break
                    else:
                        raise RuntimeError("Contour goes outisde the bounds. "
                                           "Set 'open_contours' to True if "
                                           "that is the expected behavior.")

                if ij == initial_point:
                    # The contour closes on itself
                    contours_cells.append(single_contour)
                    contour_paths.append(single_path)
                    break

        return contours_cells, contour_paths


def marching_cell_values(ij, d_ij, grid_values, x_array, y_array, level=0, mod=None):
    """
    Return the interpolated values where the contour crosses
    the new boundary between squares.

    Parameters
    ----------
    ij : pair of ints
        Zeroth position of the cell on the grid.
    d_ij : pair of ints
        Direction where the marching cell is moving.
    grid_values: ndarray of shape `(n, m)`
        Values to use in the linear interpolation.
    level : float, default to '0'
        The level of the isosurface, that is, contour in 2d.
    mod: paif of ints
        If
    """
    if mod is None:
        mod = (2**32, 2**32)  # very large ints (no overflow in python 3)

    def values(i, j):
        return grid_values[i % mod[0], j % mod[1]]

    i, j = ij
    if d_ij[0] == 1:
        # o-----o-----x
        # | old | new |
        # o-----o-----x
        new_values = (
            values(i+1, j),
            values(i+1, j+1),
        )
        x = x_array[i+1]
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        y = y_array[j] + weight * (y_array[j+1] - y_array[j])
    elif d_ij[0] == -1:
        # x-----o-----o
        # | new | old |
        # x-----o-----o
        new_values = (
            values(i, j),
            values(i, j+1),
        )
        x = x_array[i]
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        y = y_array[j] + weight * (y_array[j+1] - y_array[j])
    elif d_ij[1] == 1:
        # x-----x
        # | new |
        # o-----o
        # | old |
        # o-----o
        new_values = (
            values(i, j+1),
            values(i+1, j+1),
        )
        weight = (level - new_values[0]) / (new_values[1] - new_values[0])
        x = x_array[i] + weight * (x_array[i+1] - x_array[i])
        y = y_array[j+1]
    elif d_ij[1] == -1:
        # o-----o
        # | old |
        # o-----o
        # | new |
        # x-----x
        new_values = (
            values(i+1, j),
            values(i, j),
        )
        weight = (level - new_values[1]) / (new_values[0] - new_values[1])
        x = x_array[i] + weight * (x_array[i+1] - x_array[i])
        y = y_array[j]
    else:
        raise ValueError(f"The displacement {d_ij} is trivial.")

    return x, y


def marching_step(cell, func, middle, d_ij):
    """Return the direction to the next cell


    The parameters `func`, `middle` and `d_ij` are only accessed
    when they are necessary to resolve saddle-point ambiguities
    (i.e. ``cell == 0b0101`` or ``cell == 0b1010``).
    """
    try:
        return MARCHING_STEPS[cell]
    except KeyError:
        msg = "Inconsistent direction and cell values."
        new_d_ij = None
        if cell == 0b0101:
            if func(*middle) < 0:
                if d_ij == (0, 1):
                    new_d_ij = tuple([1, 0])
                elif d_ij == (0, -1):
                    new_d_ij = tuple([-1, 0])
                raise RuntimeError(msg)
            else:
                if d_ij == (0, 1):
                    new_d_ij = tuple([-1, 0])
                elif d_ij == (0, -1):
                    new_d_ij = tuple([1, 0])
                raise RuntimeError(msg)
        elif cell == 0b1010:
            if func(*middle) < 0:
                if d_ij == (1, 0):
                    new_d_ij = tuple([0, -1])
                elif d_ij == (-1, 0):
                    new_d_ij = tuple([0, 1])
                raise RuntimeError(msg)
            else:
                if d_ij == (1, 0):
                    new_d_ij = tuple([0, 1])
                elif d_ij == (-1, 0):
                    new_d_ij = tuple([0, -1])
                raise RuntimeError(msg)
        else:
            raise RuntimeError("cell " + bin(cell) + " shouldn't happen ...")

        return new_d_ij


# tables definition
""" MARCHING_STEPS : constant dict
    Assigns a direction to the square corner values.
"""
MARCHING_STEPS = {
    0b0000: tuple([1, 0]),
    0b0001: tuple([-1, 0]),
    0b0100: tuple([1, 0]),
    0b0010: tuple([0, -1]),
    0b1000: tuple([0, 1]),
    0b0011: tuple([-1, 0]),
    0b0110: tuple([0, -1]),
    0b1100: tuple([1, 0]),
    0b1001: tuple([0, 1]),
    0b0111: tuple([-1, 0]),
    0b1110: tuple([0, -1]),
    0b1101: tuple([1, 0]),
    0b1011: tuple([0, 1])
}
