(installation_chapter)=
# Installation

There are several ways that OpenFF Recharge and its dependencies can be installed, including using the `conda` 
package manager.

## Using conda

The recommended way to install `openff-recharge` is via the `conda` package manager:

```shell
conda install -c conda-forge openff-recharge
```

### Optional dependencies

#### OpenEye toolkits

If you have access to the OpenEye toolkits (namely `oechem`, `oequacpac` and `oeomega`) we recommend installing
these also as these can speed up certain operations significantly.

```shell
conda install -c openeye openeye-toolkits
```

#### Psi4

Psi4 is an open source quantum chemistry package that enables OpenFF Recharge to generate electrostatic potential and 
electric field data:

```shell
conda install -c conda-forge -c defaults -c psi4 psi4
```

#### [QCPortal](https://github.com/MolSSI/QCPortal)

QCPortal enables the retrieval of QC calculations from a running QCFractal instance (such as the 
[QCArchive](https://qcarchive.molssi.org/)) from which electrostatic potential and electric field data can be 
reconstructed.

```shell
conda install -c conda-forge qcportal
```

## From source

To install `openff-recharge` from source begin by cloning the repository from 
[github](https://github.com/openforcefield/openff-recharge),

```shell
git clone https://github.com/openforcefield/openff-recharge
cd openff-recharge
```

create a custom conda environment which contains the required dependencies and activate it,

```shell
conda env create --name openff-recharge --file devtools/conda-envs/test-env.yaml
conda activate openff-recharge
```
and finally install the package itself:

```shell
python setup.py develop
```
