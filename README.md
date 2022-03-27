# OpenFF Recharge

[![tests](https://github.com/openforcefield/openff-recharge/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/openforcefield/openff-recharge/actions/workflows/ci.yaml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/openforcefield/openff-recharge.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/openforcefield/openff-recharge/context:python)
[![codecov](https://codecov.io/gh/openforcefield/openff-recharge/branch/main/graph/badge.svg)](https://codecov.io/gh/openforcefield/openff-recharge/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenFF Recharge aims to provide a comprehensive suite of tools for training the partial charges of molecules against 
quantum chemical electrostatic potential (ESP) and electric field data.

A focus is given to training 'charge-correction models' similar to the popular AM1BCC charge model, but support for
other methods such as deriving RESP charges or training virtual sites on top of existing partial charges is also 
supported.

***Warning** - although a significant effort has been made to ensure the scientific validity of this framework 
(especially the hand-converted AM1BCC parameters), it is still under heavy development and much care should be taken 
when using it in production work.*

## Getting Started

To start using this framework we recommend looking over [the documentation](https://openff-recharge.readthedocs.io/en/latest/index.html),
especially the [installation](https://openff-recharge.readthedocs.io/en/latest/getting-started/installation.html) and 
[quick start](https://openff-recharge.readthedocs.io/en/latest/getting-started/quick-start.html) guides.

## Features

The framework currently supports:

* **Generating QC ESP and electric field data**
  * directly by interfacing with the [Psi4](https://psicode.org/) quantum chemical code
  * from wavefunctions stored within a QCFractal instance, including the [QCArchive](https://qcarchive.molssi.org/)
  
* **Defining new charge models that contain**
  * base QC (e.g. AM1 charges) or tabulated library / RESP charges
  * bond-charge corrections
  * virtual sites

* **A SMARTS port of the AM1BCC charge model**

* **Generating RESP charges for multi-conformer molecules**

* **Training charge(-correction) parameters by**
  * the normal linear least squares method (fixed v-site geometries only)
  * gradient descent using [pytorch](https://pytorch.org/) or numpy
  * Bayesian methods using frameworks like [pyro](https://pyro.ai/)
  
## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2020, Open Force Field Consortium
