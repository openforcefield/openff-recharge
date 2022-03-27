# OpenFF Recharge

OpenFF Recharge is a framework which aims to provide utilities for optimising the partial charges of molecules against 
QC derived electrostatic potential and electric field data.
 
Currently, the framework focuses on the optimization of bond-charge correction (BCC) and virtual site parameters.

:::{warning}
Although a significant effort has been made to ensure the scientific validity of this framework (especially the 
hand-converted AM1BCC parameters), it is still under heavy development and much care should be taken when using it 
in production work.

We are always looking to improve this framework so if you do find any undesirable or irritating behaviour, please 
[file an issue!]
:::

[file an issue!]: https://github.com/openforcefield/openff-recharge/issues/new/choose

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
