from collections import namedtuple
from typing import Literal
import numpy as np
from nessai.livepoint import empty_structured_array
from nessai.reparameterisations import (
    Reparameterisation,
)

from .utils import determine_parameter_name
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



class PhasePolarizationFolding(Reparameterisation):
    """Reparameterisation that folds the phase and polarization parameters.
    
    Space is folded such that the phase is in [0, pi] or [0, pi/2] and the polarization is in
    [0, pi/2].
    
    Parameters
    ----------
    parameters : list[str]
        List of parameter names. Must include a phase and polarization
        parameter.
    prior_bounds : dict[str, list]
        Dictionary of parameter bounds. Must include bounds for the phase and
        polarization parameters.
    phase_parameter : str, optional
        Name of the phase parameter. If None, will be automatically determined
        from the parameter names.
    n_phase_modes : int, optional
        Number of phase modes to fold. Must be 2 or 4. Default is 2, which folds
        the phase into [0, pi]. If 4, folds the phase into [0, pi/2].
    polarization_parameter : str, optional
        Name of the polarization parameter. If None, will be automatically
        determined from the parameter names.
    shift_mean : bool, optional
        Whether to roll the folded distribution in each dimension such that
        angular mean is at the centre of the folded space. Default is True.
    """
    
    known_phase_parameters: frozenset[str] = frozenset(["phase", "phi", "phi_ref"])
    known_polarization_parameters: frozenset[str] = frozenset(["psi", "polarization"])
    
    one_to_one: bool = False
    requires_bounded_prior: bool = True
    
    _phase_parameter: str = None
    _polarization_parameter: str = None
    
    def __init__(
        self, parameters: list[str] = None,
        prior_bounds: dict[str, list] = None,
        phase_parameter: str = None,
        n_phase_folds: Literal[1, 2, 4] = 2,
        n_polarization_folds: Literal[1, 2] = 2,
        polarization_parameter: str = None,
        rng: np.random.Generator = None,
        roll: bool = False,
    ):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds, rng=rng)
        
        self.phase_parameter = phase_parameter
        self.polarization_parameter = polarization_parameter
        self.prime_parameters = [f"{p}_folded" for p in self.parameters]
        
        if n_phase_folds not in (1, 2, 4):
            raise ValueError(f"n_phase_folds must be 1, 2 or 4. Received {n_phase_folds}")
        
        if n_polarization_folds not in (1, 2):
            raise ValueError(f"n_polarization_folds must be 1 or 2. Received {n_polarization_folds}")
        
        self.n_phase_folds = n_phase_folds
        self.phase_span = 2 * np.pi / self.n_phase_folds
        self.n_polarization_folds = n_polarization_folds
        self.polarization_span = np.pi / self.n_polarization_folds
        self.n_modes = self.n_phase_folds * self.n_polarization_folds
        self.roll_mean = roll
        self.phase_shift = 0.0
        self.polarization_shift = 0.0
        
        self.phase_mode_weights = np.ones(self.n_phase_folds) / self.n_phase_folds
        self.polarization_mode_weights = np.ones(self.n_polarization_folds) / self.n_polarization_folds
        
    @property
    def phase_parameter(self) -> str:
        return self._phase_parameter
    
    @phase_parameter.setter
    def phase_parameter(self, name: str | None):
        if name is None:
            name = determine_parameter_name(
                self.parameters, self.known_phase_parameters, required=True
            )
            logger.debug(f"Automatically determined phase parameter: {name}")
        elif name not in self.parameters:
            raise ValueError(f"Phase parameter {name} not found in parameters.")
            
        if not np.isclose(np.ptp(self.prior_bounds[name]), 2 * np.pi):
            raise ValueError(
                f"Phase parameter {name} does not span 2 pi. "
                f"Received bounds: {self.prior_bounds[name]}"
            )
        self._phase_parameter = name
        
    @property
    def polarization_parameter(self) -> str:
        return self._polarization_parameter
    
    @polarization_parameter.setter
    def polarization_parameter(self, name: str | None):
        if name is None:
            name = determine_parameter_name(
                self.parameters, self.known_polarization_parameters, required=True
            )
            logger.debug(f"Automatically determined polarization parameter: {name}")
        elif name not in self.parameters:
            raise ValueError(f"Polarization parameter {name} not found in parameters.")
        
        if not np.isclose(np.ptp(self.prior_bounds[name]), np.pi, atol=1e-3):
            raise ValueError(
                f"Polarization parameter {name} does not span pi. "
                f"Received bounds: {self.prior_bounds[name]}"
            )
        self._polarization_parameter = name
        
    @property
    def phase_parameter_folded(self) -> str:
        return f"{self.phase_parameter}_folded"
    
    @property
    def polarization_parameter_folded(self) -> str:
        return f"{self.polarization_parameter}_folded"
    
    def sample_phase_mode_index(self, size: int) -> np.ndarray:
        return self.rng.choice(self.n_phase_folds, size=size, p=self.phase_mode_weights)
    
    def sample_polarization_mode_index(self, size: int) -> np.ndarray:
        return self.rng.choice(
            self.n_polarization_folds, size=size, p=self.polarization_mode_weights
        )
        
    def calculate_shift(self, x, lower, upper):
        angles = 2 * np.pi * (x - lower) / (upper - lower)
        mean_complex = np.mean(np.exp(1j * angles))
        mean_angle = np.angle(mean_complex)
        shift = (lower + upper) / 2 - mean_angle * (upper - lower) / (2 * np.pi)
        return shift
        
    def update(self, x):
        x_prime = empty_structured_array(x.shape[0], names=self.prime_parameters)
        x, x_prime, log_j = self.fold(x, x_prime, np.zeros(x.shape[0]))
        if self.roll_mean:
            self.phase_shift = self.calculate_shift(x_prime[self.phase_parameter_folded], 0, self.phase_span)
            self.polarization_shift = self.calculate_shift(x_prime[self.polarization_parameter_folded], 0, self.polarization_span)
            logger.debug(f"Calculated phase shift: {self.phase_shift}")
            logger.debug(f"Calculated polarization shift: {self.polarization_shift}")
    
    def fold(self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray) -> np.ndarray:
        x_prime[self.phase_parameter_folded] = np.mod(x[self.phase_parameter], self.phase_span)
        x_prime[self.polarization_parameter_folded] = np.mod(
            x[self.polarization_parameter], self.polarization_span
        )
        return x, x_prime, log_j
    
    def roll(self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray) -> np.ndarray:
       x_prime[self.phase_parameter_folded] = np.mod(
            x_prime[self.phase_parameter_folded] + self.phase_shift, self.phase_span
        ) 
       x_prime[self.polarization_parameter_folded] = np.mod(
            x_prime[self.polarization_parameter_folded] + self.polarization_shift,
            self.polarization_span,
        )
       return x, x_prime, log_j
    
    def unfold(self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray) -> np.ndarray:
        """Unfold"""
        phase_mode_index = self.sample_phase_mode_index(size=x.shape[0])
        polarization_mode_index = self.sample_polarization_mode_index(size=x.shape[0])
        x[self.phase_parameter] = (
            x_prime[self.phase_parameter_folded]
            + phase_mode_index * self.phase_span
        )
        x[self.polarization_parameter] = (
            x_prime[self.polarization_parameter_folded]
            + polarization_mode_index * self.polarization_span
        )
        return x, x_prime, log_j
    
    def unroll(self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray) -> np.ndarray:
        x_prime[self.phase_parameter_folded] = np.mod(
            x_prime[self.phase_parameter_folded] - self.phase_shift, self.phase_span
        )
        x_prime[self.polarization_parameter_folded] = np.mod(
            x_prime[self.polarization_parameter_folded] - self.polarization_shift,
            self.polarization_span,
        )
        return x, x_prime, log_j
    
    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x, x_prime, log_j = self.fold(x, x_prime, log_j)
        if self.roll_mean:
            x, x_prime, log_j = self.roll(x, x_prime, log_j)
        return x, x_prime, log_j
    
    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        if self.roll_mean:
            x, x_prime, log_j = self.unroll(x, x_prime, log_j)
        return self.unfold(x, x_prime, log_j)