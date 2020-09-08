import click

from openff.recharge.cli.generate import generate


@click.group()
def cli():
    """The root CLI group for all ``openff-recharge`` commands"""


cli.add_command(generate)
