# Release History

Releases follow the `major.minor.micro` scheme recommended by [PEP440](https://www.python.org/dev/peps/pep-0440/#final-releases), where

* `major` increments denote a change that may break API compatibility with previous `major` releases
* `minor` increments add features but do not break API compatibility
* `micro` increments represent bugfix releases or improvements in documentation

## Current development

## 0.5.1

* #141: Update badges
* #143: Pin to legacy QCArchive software
* #145: Update versioneer for Python 3.12 compatibility
* #147: Avoid internally throwing `AtomMappingWarning`
* #148: Stage Pydantic v2 compatibility
* #150: Move tests modules to private API

## 0.5.0

This release re-introduces virtual site support and is only compatible with OpenFF Toolkit versions 0.11.0 and newer.
