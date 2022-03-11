(cli_chapter)=
# Commands

OpenFF Recharge provides a limited command line interface to its major features under the alias `openff-recharge`:

```sh
openff-recharge --help
openff-recharge reconstruct --help
```

See the [quick start](quick_start_chapter) guide for examples of using the CLI.

(cli_ref)=
<!--
The click directive renders to rST,
so we must use eval-rst here
-->
:::{eval-rst}
.. click:: openff.recharge.cli:cli
    :prog: openff-recharge
    :nested: full
:::
