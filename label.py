import argparse
import os

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

from sbc.calculator import MACECalculator


def optimize(atoms, fixed_cell=True, fmax=1e-3, steps=2000):
    if atoms.pbc.any() and not fixed_cell:
        dof = FrechetCellFilter(
            atoms,
            mask=[True] * 6,
        )
    else:
        dof = atoms

    trajectory = []

    optimizer = BFGS(dof, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)


if __name__ == '__main__':
    from order_parameter import get_phase
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--nclients",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--xyz",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--fixed_cell",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    data = read(args.xyz, index=':')
    atoms = data[0].copy()
    model_calculator = MACECalculator(
        model_path=args.model,
        device='cuda',
        dtype='float64',
    )
    atoms.calc = model_calculator

    if args.start == -1:  # not passed, assume 0 unless SLURM variable is available
        slurm_procid = os.environ.get('SLURM_PROCID', None)
        if slurm_procid is None:  # just start at 0
            i = 0
        else:
            assert slurm_procid < args.nclients
            i = slurm_procid

    labeled = []
    while i < len(data):
        print('optimizing state {}'.format(i))
        state = data[i]
        atoms.set_positions(state.get_positions())
        atoms.set_cell(state.get_cell())
        optimize(atoms, args.fixed_cell, args.fmax, args.max_steps)
        e = atoms.calc.results['energy']

        phase, meta = get_phase(atoms)
        print('state {}  |  E = {:10.4f}  |  {} ({})'.format(i, e, phase, meta))

        if phase is not None:
            state.info['phase'] = phase
        labeled.append(state)

        i += args.nclients

    write('labeled_{}.xyz'.format(args.start), labeled)
