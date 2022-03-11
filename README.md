# OpenFF Recharge

[![tests](https://github.com/openforcefield/openff-recharge/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/openforcefield/openff-recharge/actions/workflows/ci.yaml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-recharge.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-recharge/context:python)
[![codecov](https://codecov.io/gh/openforcefield/openff-recharge/branch/main/graph/badge.svg)](https://codecov.io/gh/openforcefield/openff-recharge/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenFF Recharge is a framework which aims to provide utilities for optimising the partial charges of molecules against 
QC derived electrostatic potential and electric field data.
 
Currently, the framework focuses on the optimization of bond-charge correction (BCC) and virtual site parameters.

***Warning** - although a significant effort has been made to ensure the scientific validity of this framework 
(especially the hand-converted AM1BCC parameters), it is still under heavy development and much care should be taken 
when using it in production work.*

## Getting Started

See the [examples](examples) directory for example scripts to get started with.

## Installation

### Conda installation

The base `openff-recharge` package can be installed using conda as follows:

```shell
conda install -c conda-forge openff-recharge
```

If you have access to the OpenEye cheminformatics toolkits we recommend also installing these to speed up 
certain features such as charge and conformer generation:

```shell
conda install -c openeye openeye-toolkits
``` 

***Note:** The OpenEye dependency will be removed in future versions of the framework.*

### Optional dependencies

To make the full use of the framework, including the computation of ESP / electric field data, it is recommended to 
install the following optional dependencies 

#### [Psi4](http://www.psicode.org/)

```
# (OPTIONAL) Enable ESP and field computation and reconstruction.
conda install -c defaults -c psi4 "psi4 >=1.4"
```

#### [QCPortal](https://github.com/MolSSI/QCPortal)

```
# (OPTIONAL) Enable reconstructing ESP and field data from QCArchive.
conda install -c conda-forge qcportal
```

## Copyright

Copyright (c) 2020, Open Force Field Consortium
