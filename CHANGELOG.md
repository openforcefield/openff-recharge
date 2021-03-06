# Changelog

## [0.0.1-alpha.6](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.6) (2020-12-04)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.1-alpha.5...0.0.1-alpha.6)

**Implemented enhancements:**

- Add CLI to Reconstruct the ESP / EF from QCA Results [\#55](https://github.com/openforcefield/openff-recharge/pull/55) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.5](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.5) (2020-11-30)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.1-alpha.4...0.0.1-alpha.5)

**Implemented enhancements:**

- Basic ESP Store Provenance [\#53](https://github.com/openforcefield/openff-recharge/pull/53) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Compute ESP Data from QCFractal Results [\#52](https://github.com/openforcefield/openff-recharge/pull/52) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add a Requires Package Utility Decorator [\#49](https://github.com/openforcefield/openff-recharge/pull/49) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- First Pass at Sulfur BCC Parameters [\#31](https://github.com/openforcefield/openff-recharge/pull/31) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

**Merged pull requests:**

- Automatically Remove Orphaned Provenance [\#54](https://github.com/openforcefield/openff-recharge/pull/54) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.4](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.4) (2020-09-22)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.1-alpha.3...0.0.1-alpha.4)

**Implemented enhancements:**

- Expose an Option to Control the PSI4 DFT Grid [\#47](https://github.com/openforcefield/openff-recharge/pull/47) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.3](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.3) (2020-09-18)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.1-alpha.2...0.0.1-alpha.3)

**Fixed bugs:**

- Catch Generate CLI Exceptions [\#45](https://github.com/openforcefield/openff-recharge/pull/45) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.2](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.2) (2020-09-08)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.1-alpha.1...0.0.1-alpha.2)

**Implemented enhancements:**

- Add a CLI to Generate ESP from Schema Settings [\#43](https://github.com/openforcefield/openff-recharge/pull/43) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Make Package Importable Without OE Installed [\#42](https://github.com/openforcefield/openff-recharge/pull/42) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add Mechanisms to Train Against Electric Field Data [\#39](https://github.com/openforcefield/openff-recharge/pull/39) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Store Electric Field [\#38](https://github.com/openforcefield/openff-recharge/pull/38) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

**Fixed bugs:**

- Fix Psi4 SEGFAULT on OSX CI [\#41](https://github.com/openforcefield/openff-recharge/pull/41) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Return Applied Corrections In Original Order [\#40](https://github.com/openforcefield/openff-recharge/pull/40) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.1](https://github.com/openforcefield/openff-recharge/tree/0.0.1-alpha.1) (2020-08-10)

[Full Changelog](https://github.com/openforcefield/openff-recharge/compare/0.0.0...0.0.1-alpha.1)

**Implemented enhancements:**

- Replace Decimal Type in ESP Store with Float Stored as Int [\#36](https://github.com/openforcefield/openff-recharge/pull/36) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Adds PCM support for ESP calculations. [\#35](https://github.com/openforcefield/openff-recharge/pull/35) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Standardise Conformer Generation to use Factory + Settings Pattern [\#34](https://github.com/openforcefield/openff-recharge/pull/34) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Clean-up Optimisation Help Class [\#33](https://github.com/openforcefield/openff-recharge/pull/33) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add SMIRNOFF Support [\#30](https://github.com/openforcefield/openff-recharge/pull/30) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Rename BCC Classes to be More Obvious and Consistent [\#28](https://github.com/openforcefield/openff-recharge/pull/28) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Adds Utility to Re-order Geometry to Match a Tagged Molecule [\#25](https://github.com/openforcefield/openff-recharge/pull/25) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Generalise `BCCGenerator.applied\_corrections` [\#24](https://github.com/openforcefield/openff-recharge/pull/24) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Rename inverse\_distance\_matrix [\#21](https://github.com/openforcefield/openff-recharge/pull/21) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Standardise Syntax of ESPGenerator [\#20](https://github.com/openforcefield/openff-recharge/pull/20) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add Inverse Distance Matrix Utility [\#19](https://github.com/openforcefield/openff-recharge/pull/19) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add SQLite Store for Calculated ESP Data [\#18](https://github.com/openforcefield/openff-recharge/pull/18) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Expose the AM1 Optimize and Symmetrize Options [\#12](https://github.com/openforcefield/openff-recharge/pull/12) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Guess Stereochemistry from SMILES [\#11](https://github.com/openforcefield/openff-recharge/pull/11) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Expand BCC Smirks to Nitrogen and Halogens [\#10](https://github.com/openforcefield/openff-recharge/pull/10) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Refactor the Charge Generator Classes [\#9](https://github.com/openforcefield/openff-recharge/pull/9) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Rename the `generators` Module to `charges` [\#8](https://github.com/openforcefield/openff-recharge/pull/8) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add a Psi4 Implementation for Computing Electrostatic Potentials [\#6](https://github.com/openforcefield/openff-recharge/pull/6) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add Electrostatic Potential Base Classes [\#5](https://github.com/openforcefield/openff-recharge/pull/5) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add Temporary Change Directory Context Manager [\#4](https://github.com/openforcefield/openff-recharge/pull/4) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add a Utility to Extract a Molecules Conformers [\#2](https://github.com/openforcefield/openff-recharge/pull/2) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Initial Grid Generator Implementation [\#1](https://github.com/openforcefield/openff-recharge/pull/1) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

**Fixed bugs:**

- Validate PCM Cavity Area is Positive [\#37](https://github.com/openforcefield/openff-recharge/pull/37) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Remove BCC Parameters Duplicated by Aromatic Bond Type [\#29](https://github.com/openforcefield/openff-recharge/pull/29) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Re-enable Dense Conformer Generation [\#27](https://github.com/openforcefield/openff-recharge/pull/27) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Fix AM1BCC Aromaticity for Fused Aromatic Rings [\#17](https://github.com/openforcefield/openff-recharge/pull/17) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Fix AM1BCC Aromaticity Model for Acenaphthene [\#15](https://github.com/openforcefield/openff-recharge/pull/15) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Generate Lattice Grid Inside Rectangular Box [\#3](https://github.com/openforcefield/openff-recharge/pull/3) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

**Merged pull requests:**

- Initial BCC Optimization Utilities and Scripts [\#26](https://github.com/openforcefield/openff-recharge/pull/26) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Clean-up the Validation Script [\#16](https://github.com/openforcefield/openff-recharge/pull/16) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Move the Parity Checks to a Script [\#14](https://github.com/openforcefield/openff-recharge/pull/14) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add AM1BCC Validation Utility [\#13](https://github.com/openforcefield/openff-recharge/pull/13) ([SimonBoothroyd](https://github.com/SimonBoothroyd))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
