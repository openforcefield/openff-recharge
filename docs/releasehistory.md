# Release History

Releases follow the `major.minor.micro` scheme recommended by [PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

## 0.5.3

* #166: Drop Python 3.9
* #170: Add memory specification to file
* #172: Expand bondi radii
* #174: Use new Pydantic v1 backdoor
* #177: Use multiprocessing when rebuilding ESPs

## 0.5.2

* #149: Update tests to use "new" QCArchive stack
* #158: Update qcportal results with solvent
* #159: Regenerate MSK grid with new OEChem
* #162: Fix Psi4 always minimize with SCF
* #161: Fix multi-conformer RESP restraint strength

## 0.5.1

* #141: Update badges
* #143: Pin to legacy QCArchive software
* #145: Update versioneer for Python 3.12 compatibility
* #147: Avoid internally throwing `AtomMappingWarning`
* #148: Stage Pydantic v2 compatibility
* #150: Move tests modules to private API
* #151: Optionally generate Psi4 ESPs with multiple threads

## 0.5.0

This release re-introduces virtual site support and is only compatible with OpenFF Toolkit versions 0.11.0 and newer.
