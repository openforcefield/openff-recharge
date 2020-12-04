import click

from openff.recharge.cli.generate import generate
from openff.recharge.cli.reconstruct import reconstruct


@click.group()
def cli():
    """The root CLI group for all ``openff-recharge`` commands"""


cli.add_command(generate)
cli.add_command(reconstruct)
