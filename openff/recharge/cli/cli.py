import click

from openff.recharge.cli.generate import generate


@click.group()
def cli():
    pass


cli.add_command(generate)
