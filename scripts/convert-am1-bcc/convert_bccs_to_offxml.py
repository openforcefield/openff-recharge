import click

from openff.toolkit import ForceField
from openff.units import unit
from openff.recharge.charges.bcc import BCCCollection


@click.command()
@click.option(
    "--input-file",
    default="openeye-am1-bcc.json",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Input BCC collection (JSON)"
)
@click.option(
    "--output-file",
    default="openeye-am1-bcc.offxml",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="Output BCC force field (offxml)"
)
@click.option(
    "--partial-charge-method",
    default="AM1-Mulliken",
    type=click.Choice(["zeros", "AM1-Mulliken"]),
    help=(
        "Charge method to use in combination with the BCCs. "
        "Use 'zeros' for debugging, 'AM1-Mulliken' for production."
    )
)
def convert_bccs_to_offxml(
    input_file,
    output_file,
    partial_charge_method,
):
    collection = BCCCollection.parse_file(input_file)
    ff = ForceField()

    ff.get_parameter_handler("Electrostatics")
    handler = ff.get_parameter_handler("ChargeIncrementModel")
    handler.partial_charge_method = partial_charge_method

    for parameter in collection.parameters[::-1]:
        handler.add_parameter({
            "smirks": parameter.smirks,
            "charge_increment": [parameter.value * unit.elementary_charge],
            "id": parameter.provenance["code"]
        })

    ff.to_file(output_file)


if __name__ == "__main__":
    convert_bccs_to_offxml()
