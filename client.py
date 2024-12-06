import argparse
from pathlib import Path
import numpy as np

from ase import Atoms
from ase.io import read
from ase.calculators.calculator import Calculator

from ipi._driver.driver import run_driver
from ipi.utils.units import unit_to_user, unit_to_internal

from sbc.calculator import MACECalculator, MetadynamicsCalculator


class Driver:

    def __init__(
        self,
        atoms: Atoms,
        calculator: Calculator,
    ):
        self.atoms = atoms
        self.calculator = calculator

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the ASE calculator
        This routine assumes that the client will take positions
        in angstrom, and return energies in electronvolt, and forces
        in ev/ang.
        """

        # ASE calculators assume angstrom and eV units
        pos = unit_to_user("length", "angstrom", pos)

        # ASE expects cell-vectors-as-rows
        cell = unit_to_user("length", "angstrom", cell.T)
        # applies the cell and positions to the template
        structure = self.atoms.copy()
        structure.positions[:] = pos
        structure.cell[:] = cell
        structure.calc = self.calculator

        # Do the actual calculation
        #properties = structure.get_properties(["energy", "forces", "stress"])
        pot = structure.get_potential_energy()
        force = structure.get_forces()
        stress = structure.get_stress()
        if len(stress) == 6:
            # converts from voight notation
            stress = np.array(stress[[0, 5, 4, 5, 1, 3, 4, 3, 2]])

        # converts to internal quantities
        pot_ipi = np.asarray(
            unit_to_internal("energy", "electronvolt", pot), np.float64
        )
        force_ipi = np.asarray(unit_to_internal("force", "ev/ang", force), np.float64)
        vir_calc = -stress * structure.get_volume()
        vir_ipi = np.array(
            unit_to_internal("energy", "electronvolt", vir_calc.T), dtype=np.float64
        )
        extras = ""

        return pot_ipi, force_ipi, vir_ipi, extras


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MACE calculator with socket communication')

    parser.add_argument('--xyz',
                        type=str,
                        required=True,
                        help='Path to XYZ file to initialize calculator')

    parser.add_argument('--mode',
                        type=str,
                        choices=['energy', 'hills'],
                        required=True,
                        help='Calculator mode: energy or hills')

    parser.add_argument('--address',
                        type=str,
                        required=True,
                        help='socket')

    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Device to run on: cpu, cuda:0, cuda:1, etc.')

    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to PyTorch model file')
    parser.add_argument('--hills',
                        type=str,
                        default=None,
                        required=False,
                        help='Path to PyTorch model file')
    parser.add_argument('--height',
                        type=float,
                        default=0.05,  # ~ 5 kJ/mol
                        required=False,
                        help='Path to PyTorch model file')
    parser.add_argument('--sigma',
                        type=float,
                        default=1,
                        required=False,
                        help='Path to PyTorch model file')
    parser.add_argument('--frequency',
                        type=int,
                        default=100,
                        required=False,
                        help='Path to PyTorch model file')
    parser.add_argument('--warmup',
                        action="store_true",
                        help='Path to XYZ file to initialize calculator')

    return parser.parse_args()


def main():
    args = parse_arguments()
    atoms = read(args.xyz)

    # Initialize calculator with specified parameters
    if args.mode == 'energy':
        calculator = MACECalculator(
            model_path=args.model_path,
            device=args.device,
            default_dtype='float32',
        )
    elif args.mode == 'hills':
        calculator = MetadynamicsCalculator(
            model_path=args.model_path,
            device=args.device,
            default_dtype='float32',
            frequency=args.frequency,
            height=args.height,
            sigma=args.sigma,
            path_hills=Path(args.hills)
        )

    atoms.calc = calculator
    if args.warmup:
        for i in range(10):
            atoms.get_potential_energy()
            atoms.calc.reset()

    # Set up socket connection
    driver = Driver(atoms, calculator)
    run_driver(
        unix=True,
        address=str(Path.cwd() / args.address),
        driver=driver,
        sockets_prefix="",
    )


if __name__ == '__main__':
    main()
