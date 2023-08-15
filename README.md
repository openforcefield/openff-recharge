# OpenFF Recharge

| **Test status** | [![CI Status](https://github.com/openforcefield/openff-recharge/workflows/ci/badge.svg)](https://github.com/openforcefield/openff-recharge/actions?query=branch%3Amain+workflow%3Aci) | [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/openforcefield/openff-recharge/main.svg)](https://results.pre-commit.ci/latest/github/openforcefield/openff-recharge/main) |
|:-|:-|:-|
| **Code quality** | [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) | [![Codecov coverage](https://img.shields.io/codecov/c/github/openforcefield/openff-recharge.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/openforcefield/openff-recharge)
| **Latest release** | ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/openforcefield/openff-recharge?include_prereleases)
| **User support** | [![Documentation Status](https://readthedocs.org/projects/openff-recharge/badge/?version=latest)](https://openff-recharge.readthedocs.io/en/latest/?badge=latest) | [![Discussions](https://img.shields.io/badge/Discussions-GitHub-blue?logo=github)](https://github.com/openforcefield/discussions/discussions)

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

## How to Cite

Please cite OpenFF Recharge using the [Zenodo record](https://zenodo.org/record/8118623) of the [latest release](https://zenodo.org/record/8118623) or the version that was used. The BibTeX reference of the latest release can be found [here](https://zenodo.org/record/8118623/export/hx).

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
