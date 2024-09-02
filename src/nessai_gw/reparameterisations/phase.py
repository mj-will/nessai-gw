from nessai.reparameterisations import (
    Reparameterisation,
)
import numpy as np

from .. import nessai_logger


logger = nessai_logger.getChild(__name__)


class DeltaPhaseReparameterisation(Reparameterisation):
    """Reparameterisation that converts phase to delta phase.

    The Jacobian determinant of this transformation is 1.

    Requires "psi" and "theta_jn".

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name(s) of the parameter(s).
    prior_bounds : Union[list, dict]
        Prior bounds for the parameters
    """

    def __init__(self, parameters=None, prior_bounds=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.requires = ["psi", "theta_jn"]
        self.prime_parameters = ["delta_phase"]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space to x'-space.

        Parameters
        ----------
        x : structured array
            Array of inputs
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        x_prime[self.prime_parameters[0]] = (
            x[self.parameters[0]] + np.sign(np.cos(x["theta_jn"])) * x["psi"]
        )
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        x[self.parameters[0]] = np.mod(
            x_prime[self.prime_parameters[0]]
            - np.sign(np.cos(x["theta_jn"])) * x["psi"],
            2 * np.pi,
        )
        return x, x_prime, log_j
