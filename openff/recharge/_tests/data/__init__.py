import os.path

import numpy
from pkg_resources import resource_filename

UNIT_CONNOLLY_SPHERE = numpy.genfromtxt(
    resource_filename(
        "openff.recharge",
        os.path.join("tests", "data", "grids", "unit-connolly-sphere.txt"),
    ),
    delimiter=" ",
)

ARGON_FCC_GRID = numpy.genfromtxt(
    resource_filename(
        "openff.recharge", os.path.join("tests", "data", "grids", "argon-fcc-grid.txt")
    ),
    delimiter=" ",
)
WATER_MSK_GRID = numpy.genfromtxt(
    resource_filename(
        "openff.recharge", os.path.join("tests", "data", "grids", "water-msk-grid.txt")
    ),
    delimiter=" ",
)
