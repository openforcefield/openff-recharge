import os
import shutil
import subprocess

from openff.toolkit.topology import Molecule


def main():

    smiles = ["C", "CC", "CCC", "CO", "CCO", "CCCO", "CCOC", "c1ccccc1", "c1occc1"]

    for i, pattern in enumerate(smiles):

        molecule: Molecule = Molecule.from_smiles(pattern)
        molecule.generate_conformers(n_conformers=1)

        input_directory = os.path.join("input", "molecules", "mol1", "conf1")
        os.makedirs(input_directory, exist_ok=True)

        shutil.copyfile("input-template.yml", os.path.join("input", "input.yml"))
        shutil.copyfile("respyte-template.yml", os.path.join("input", "respyte.yml"))

        input_path = os.path.join(input_directory, "mol1_conf1.pdb")
        molecule.to_file(input_path, "PDB")

        with open(input_path) as file:
            contents = file.read()
        with open(input_path, "w") as file:
            file.write(contents.replace("UNL", "MOL"))

        try:
            subprocess.check_call(["respyte-esp_generator"])
        except subprocess.CalledProcessError:
            shutil.rmtree("input", ignore_errors=True)
            print(f"{pattern} failed")

        shutil.move("input", f"input-{i + 1}")

        try:
            subprocess.check_call(["respyte-optimizer", f"input-{i + 1}"])
        except subprocess.CalledProcessError:
            shutil.rmtree(f"input-{i + 1}", ignore_errors=True)
            print(f"{pattern} failed")

        shutil.move("resp_output", f"output-{i + 1}")

        for file_name in ["xyz.txt", "esp.txt", "grid.txt"]:
            shutil.move(file_name, f"output-{i + 1}")


if __name__ == "__main__":
    main()
