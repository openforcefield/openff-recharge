import functools

import numpy
import pytest

from openff.recharge.charges.resp.solvers import (
    IterativeSolver,
    RESPNonLinearSolver,
    RESPSolverError,
    SciPySolver,
)


class TestRESPNonLinearSolver:
    def test_loss(self):
        loss = RESPNonLinearSolver.loss(
            beta=numpy.array([[3.0], [-3.0]]),
            design_matrix=numpy.array([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]),
            reference_values=numpy.array([[1.0], [2.0]]),
            constraint_matrix=numpy.array([[1, 1]]),
            restraint_a=6.0,
            restraint_b=4.0,
            restraint_indices=[1],
            n_conformers=1,
        )
        assert loss.shape == ()

        expected_loss = (2.0**2 + 4.0**2) + (6.0 * (5.0 - 4.0))  # chi_esp, chi_restr

        assert numpy.isclose(loss, expected_loss)

    def test_jacobian(self):
        kwargs = dict(
            design_matrix=numpy.array([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]),
            constraint_matrix=numpy.array([[1, 1]]),
            reference_values=numpy.array([[1.0], [2.0]]),
            restraint_a=6.0,
            restraint_b=4.0,
            restraint_indices=[0],
            n_conformers=1,
        )

        loss_func = functools.partial(RESPNonLinearSolver.loss, **kwargs)

        jacobian = RESPNonLinearSolver.jacobian(
            beta=numpy.array([[3.0], [-3.0]]), **kwargs
        )
        assert jacobian.shape == (2,)

        h = 0.0001

        expected_jacobian = numpy.array(
            [
                (
                    loss_func(numpy.array([[3.0 + h], [-3.0]]))
                    - loss_func(numpy.array([[3.0 - h], [-3.0]]))
                )
                / (h * 2.0),
                (
                    loss_func(numpy.array([[3.0], [-3.0 + h]]))
                    - loss_func(numpy.array([[3.0], [-3.0 - h]]))
                )
                / (h * 2.0),
            ]
        )

        assert numpy.allclose(jacobian, expected_jacobian)

    @pytest.mark.parametrize(
        "restraint_a, expected_value",
        [
            (0.0, numpy.array([[0.6], [-0.3]])),
            (1.0, numpy.array([[0.5698], [-0.2698]])),
        ],
    )
    def test_initial_guess(self, restraint_a, expected_value):
        initial_values = RESPNonLinearSolver.initial_guess(
            design_matrix=numpy.array([[1.0 / 0.3, 2.0 / 0.3], [3.0 / 0.3, 5.0 / 0.3]]),
            reference_values=numpy.array([[0.0], [1.0]]),
            constraint_matrix=numpy.array([[1, 1]]),
            constraint_values=numpy.array([[0.3]]),
            restraint_a=restraint_a,
            restraint_indices=[0, 1],
            n_conformers=1,
        )

        assert initial_values.shape == expected_value.shape
        assert numpy.allclose(initial_values, expected_value, atol=0.0001)


class TestIterativeSolver:
    def test_solve(self, monkeypatch):
        monkeypatch.setattr(
            IterativeSolver, "initial_guess", lambda *_: numpy.array([-3.5, 3.5])
        )

        charges = IterativeSolver().solve(
            design_matrix=numpy.array([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]),
            reference_values=numpy.array([[-1.0], [-2.0]]),
            constraint_matrix=numpy.array([[1.0, 1.0]]),
            constraint_values=numpy.array([[0.0]]),
            restraint_a=0.0005,
            restraint_b=0.1,
            restraint_indices=[1],
            n_conformers=1,
        )

        assert charges.shape == (2, 1)
        assert numpy.allclose(charges, numpy.array([[3.0], [-3.0]]), atol=0.001)


class TestSciPySolver:
    def test_solve(self, monkeypatch):
        monkeypatch.setattr(
            SciPySolver, "initial_guess", lambda *_: numpy.array([-3.5, 3.5])
        )

        charges = SciPySolver().solve(
            design_matrix=numpy.array([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]),
            reference_values=numpy.array([[-1.0], [-2.0]]),
            constraint_matrix=numpy.array([[1.0, 1.0]]),
            constraint_values=numpy.array([[0.0]]),
            restraint_a=0.0005,
            restraint_b=0.1,
            restraint_indices=[1],
            n_conformers=1,
        )

        assert charges.shape == (2, 1)
        assert numpy.allclose(charges, numpy.array([[3.0], [-3.0]]), atol=0.001)

    def test_solve_error(self):
        with pytest.raises(RESPSolverError, match="SciPy solver with method=SLSQP"):
            SciPySolver(method="SLSQP").solve(
                design_matrix=numpy.array(
                    [[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]
                ),
                reference_values=numpy.array([[-1.0], [-2.0]]),
                constraint_matrix=numpy.array([[1.0, 1.0], [1.0, 1.0]]),
                constraint_values=numpy.array([[0.0], [1.0]]),
                restraint_a=0.0005,
                restraint_b=0.1,
                restraint_indices=[1],
                n_conformers=1,
            )
