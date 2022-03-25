"""Different solvers for solving the non-linear RESP loss function"""
import abc
import functools
from typing import List, Literal, cast

import numpy


class RESPSolverError(RuntimeError):
    """An exception raised when a non-linear solver fails to converge."""


class RESPNonLinearSolver(abc.ABC):
    """The base for classes that will attempt to find a set of charges that minimizes
    the RESP loss function.
    """

    @classmethod
    def loss(
        cls,
        beta: numpy.ndarray,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:
        """Returns the current value of the loss function complete with restraints
        on specified charges.

        Parameters
        ----------
        beta
            The current vector of charge values with shape=(n_values,)
        design_matrix
            The design matrix that when right multiplied by ``beta`` yields the
            ESP due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference ESP values with shape=(n_grid_points, 1).
        restraint_a
            The a term in the hyperbolic RESP restraint function.
        restraint_b
            The b term in the hyperbolic RESP restraint function.
        restraint_indices
            The indices of the charges in ``beta`` that the restraint should be applied
            to.

        Returns
        -------
            The value of the loss function with shape=(1,)
        """

        beta = beta.reshape(-1, 1)

        delta = design_matrix @ beta - reference_values
        chi_esp_sqr = (delta * delta).sum()

        beta_restrained = beta[restraint_indices]
        chi_restraint_sqr = (
            restraint_a
            * (
                numpy.sqrt(
                    beta_restrained * beta_restrained + restraint_b * restraint_b
                )
                - restraint_b
            ).sum()
        )

        return chi_esp_sqr + chi_restraint_sqr

    @classmethod
    def jacobian(
        cls,
        beta: numpy.ndarray,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ):
        """Returns the jacobian of the loss function with respect to ``beta``.

        Parameters
        ----------
        beta
            The current vector of charge values with shape=(n_values,)
        design_matrix
            The design matrix that when right multiplied by ``beta`` yields the
            ESP due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference ESP values with shape=(n_grid_points, 1).
        restraint_a
            The a term in the hyperbolic RESP restraint function.
        restraint_b
            The b term in the hyperbolic RESP restraint function.
        restraint_indices
            The indices of the charges in ``beta`` that the restraint should be applied
            to.

        Returns
        -------
            The value of the jacobian with shape=(n_values,)
        """

        beta = beta.reshape(-1, 1)

        delta = design_matrix @ beta - reference_values
        d_chi_esp_sqr = 2.0 * design_matrix.T @ delta

        d_chi_restraint_sqr = restraint_a * numpy.array(
            [
                0.0
                if i not in restraint_indices
                else float(
                    beta[i] / numpy.sqrt(beta[i] * beta[i] + restraint_b * restraint_b)
                )
                for i in range(len(beta))
            ]
        )

        return d_chi_esp_sqr.flatten() + d_chi_restraint_sqr

    @classmethod
    def initial_guess(
        cls,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:
        """Compute an initial guess of the charge values by solving the lagrangian
        constrained ``Ax + b`` equations where here ``A`` does not contain any
        restraints.

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``beta`` yields the
            ESP due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference ESP values with shape=(n_grid_points, 1).
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)
        restraint_a
            The a term in the hyperbolic RESP restraint function.
        restraint_indices
            The indices of the charges in ``beta`` that the restraint should be applied
            to.

        Returns
        -------
            An initial guess of the charge values with shape=(n_values, 1)
        """

        b_matrix = (
            2.0
            * numpy.eye(design_matrix.shape[1])
            @ numpy.array(
                [
                    [restraint_a if i in restraint_indices else 0.0]
                    for i in range(design_matrix.shape[1])
                ]
            )
        )
        a_matrix = numpy.block(
            [
                [design_matrix.T @ design_matrix + b_matrix, constraint_matrix.T],
                [constraint_matrix, numpy.zeros([constraint_matrix.shape[0]] * 2)],
            ]
        )

        b_vector = numpy.vstack([design_matrix.T @ reference_values, constraint_values])

        initial_values, *_ = numpy.linalg.lstsq(a_matrix, b_vector, rcond=None)
        return initial_values[: design_matrix.shape[1]]

    @abc.abstractmethod
    def _solve(
        self,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:
        """The internal implementation of ``solve``

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``beta`` yields the
            ESP due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference ESP values with shape=(n_grid_points, 1).
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)
        restraint_a
            The a term in the hyperbolic RESP restraint function.
        restraint_b
            The b term in the hyperbolic RESP restraint function.
        restraint_indices
            The indices of the charges in ``beta`` that the restraint should be applied
            to.

        Raises
        ------
        RESPSolverError

        Returns
        -------
            The set of charge values that minimize the RESP loss function with
            shape=(n_values, 1)
        """

        raise NotImplementedError

    def solve(
        self,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:
        """Attempts to find a minimum solution to the RESP loss function.

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``beta`` yields the
            ESP due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference ESP values with shape=(n_grid_points, 1).
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)
        restraint_a
            The a term in the hyperbolic RESP restraint function.
        restraint_b
            The b term in the hyperbolic RESP restraint function.
        restraint_indices
            The indices of the charges in ``beta`` that the restraint should be applied
            to.

        Raises
        ------
        RESPSolverError

        Returns
        -------
            The set of charge values that minimize the RESP loss function with
            shape=(n_values, 1)
        """

        if design_matrix.shape[1] == 0:
            return numpy.zeros((0, 1))

        solution = self._solve(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
            restraint_a,
            restraint_b,
            restraint_indices,
        )

        predicted_total_charge = constraint_matrix @ solution
        assert predicted_total_charge.shape == constraint_values.shape

        if not numpy.allclose(predicted_total_charge, constraint_values):
            raise RESPSolverError("The total charge was not conserved by the solver")

        return cast(numpy.ndarray, solution)


class IterativeSolver(RESPNonLinearSolver):
    """Attempts to find a set of charges that minimizes the RESP loss function
    by repeated applications of the least-squares method assuming the restraints
    are linear.
    """

    @classmethod
    def _solve_iteration(
        cls,
        beta: numpy.ndarray,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ):

        b_matrix = numpy.eye(design_matrix.shape[1]) @ numpy.array(
            [
                [
                    float(
                        restraint_a
                        / numpy.sqrt(value * value + restraint_b * restraint_b)
                        if i in restraint_indices
                        else 0.0
                    )
                ]
                for i, value in enumerate(beta)
            ]
        )
        a_matrix = numpy.block(
            [
                [design_matrix.T @ design_matrix + b_matrix, constraint_matrix.T],
                [constraint_matrix, numpy.zeros([constraint_matrix.shape[0]] * 2)],
            ]
        )

        b_vector = numpy.vstack([design_matrix.T @ reference_values, constraint_values])

        beta_new, *_ = numpy.linalg.lstsq(a_matrix, b_vector, rcond=None)
        return beta_new[: design_matrix.shape[1]]

    def _solve(
        self,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:

        initial_guess = self.initial_guess(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
            restraint_a,
            restraint_indices,
        )

        iteration = 0
        tolerance = 1.0e-5

        beta_current = initial_guess

        while iteration < 300:

            beta_new = self._solve_iteration(
                beta_current,
                design_matrix,
                reference_values,
                constraint_matrix,
                constraint_values,
                restraint_a,
                restraint_b,
                restraint_indices,
            )

            beta_difference = beta_new - beta_current

            if (
                1.0
                / len(beta_new)
                * numpy.sqrt((beta_difference * beta_difference).sum())
                < tolerance
            ):
                return beta_new

            beta_current = beta_new
            iteration += 1

        raise RESPSolverError(
            "The iterative solver failed to converge after 300 iterations"
        )


class SciPySolver(RESPNonLinearSolver):
    """Attempts to find a set of charges that minimizes the RESP loss function
    using the `scipy.optimize.minimize` function.
    """

    def __init__(self, method: Literal["SLSQP", "trust-constr"] = "SLSQP"):
        """

        Parameters
        ----------
        method
            The minimizer to use.
        """

        assert method in {"SLSQP", "trust-constr"}
        self._method = method

    def _solve(
        self,
        design_matrix: numpy.ndarray,
        reference_values: numpy.ndarray,
        constraint_matrix: numpy.ndarray,
        constraint_values: numpy.ndarray,
        restraint_a: float,
        restraint_b: float,
        restraint_indices: List[int],
    ) -> numpy.ndarray:

        from scipy.optimize import LinearConstraint, minimize

        loss_function = functools.partial(
            self.loss,
            design_matrix=design_matrix,
            reference_values=reference_values,
            restraint_a=restraint_a,
            restraint_b=restraint_b,
            restraint_indices=restraint_indices,
        )
        jacobian_function = functools.partial(
            self.jacobian,
            design_matrix=design_matrix,
            reference_values=reference_values,
            restraint_a=restraint_a,
            restraint_b=restraint_b,
            restraint_indices=restraint_indices,
        )

        initial_guess = self.initial_guess(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
            restraint_a,
            restraint_indices,
        )

        # noinspection PyTypeChecker
        output = minimize(
            fun=loss_function,
            x0=initial_guess.flatten(),
            jac=jacobian_function,
            constraints=LinearConstraint(
                constraint_matrix,
                constraint_values.flatten(),
                constraint_values.flatten(),
            )
            if len(constraint_matrix) > 0
            else (),
            method=self._method,
            tol=1.0e-5,
        )

        if not output.success:

            raise RESPSolverError(
                f"SciPy solver with method={self._method} was unsuccessful: "
                f"{output.message}"
            )

        return output.x.reshape(-1, 1)
