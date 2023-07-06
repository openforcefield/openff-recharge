# OpenFF Recharge

OpenFF Recharge aims to provide a comprehensive suite of tools for training the partial charges of molecules against
quantum chemical electrostatic potential (ESP) and electric field data.

A focus is given to training 'charge-correction models' similar to the popular AM1BCC charge model, but support for
other methods such as deriving RESP charges or training virtual sites on top of existing partial charges is also
supported.

:::{warning} Although a significant effort has been made to ensure the scientific validity of this framework
(especially the hand-converted AM1BCC parameters), it is still under heavy development and much care should be taken
when using it in production work.
:::

We are always looking to improve this framework so if you do find any undesirable or irritating behaviour, please
[file an issue!]

[file an issue!]: https://github.com/openforcefield/openff-recharge/issues/new/choose

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

:::{toctree}
---
maxdepth: 2
caption: "Getting Started"
glob: True
hidden: True
---

getting-started/installation
getting-started/quick-start
getting-started/cli
releasehistory

:::

:::{toctree}
---
maxdepth: 2
caption: "User Guide"
glob: True
hidden: True
---

users/theory
users/resp

:::

<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
:::{eval-rst}
.. autosummary::
   :recursive:
   :caption: API Reference
   :toctree: ref/api
   :nosignatures:

   openff.recharge
:::
