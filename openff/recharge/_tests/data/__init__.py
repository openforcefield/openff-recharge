import os.path
from importlib.resources import files

import numpy

UNIT_CONNOLLY_SPHERE = numpy.genfromtxt(
    files(
        "openff.recharge",
    )
    / os.path.join("_tests", "data", "grids", "unit-connolly-sphere.txt"),
    delimiter=" ",
)

ARGON_FCC_GRID = numpy.genfromtxt(
    files("openff.recharge") / os.path.join("_tests", "data", "grids", "argon-fcc-grid.txt"),
    delimiter=" ",
)
WATER_MSK_GRID = numpy.genfromtxt(
    files("openff.recharge") / os.path.join("_tests", "data", "grids", "water-msk-grid.txt"),
    delimiter=" ",
)
