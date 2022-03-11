(quick_start_chapter)=
# Quick start

The OpenFF Recharge framework aims to offer a comprehensive solution for refitting different charge models to 
QC data, especially electrostatic potential (ESP) and electric field data.

The recommended way to install `openff-recharge` is via the `conda` package manager:

```shell
conda install -c conda-forge openff-recharge
```

although [several other methods are available](installation_chapter).

The fastest way to learn the framework is to look over the [examples] directory on the frameworks website. Included
are examples of how to

* generate QC electrostatic potential and electric field data using the Psi4 package and store that data in a local
  SQLite database for future use
* retrieving already available QC calculations from the [QCArchive] and reconstructing the ESP and electric field from 
  the stored wave functions

as well as

* training a new set of bond charge correction (BCC) parameters on to ESP data using `numpy`
* simultaneously training both BCC and virtual site (v-site) parameters to ESP data using `pytorch`
* performing Bayesian sampling of v-site parameters using `pyro`

[QCArchive]: https://qcarchive.molssi.org/

[examples]: https://github.com/openforcefield/openff-recharge/tree/main/examples