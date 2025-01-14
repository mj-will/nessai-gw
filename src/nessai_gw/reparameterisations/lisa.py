"""Reparameterisations for LISA"""

from collections import namedtuple
from typing import Iterable, Union

import numpy as np
from nessai.reparameterisations.base import Reparameterisation

ModeID = namedtuple("ModeID", ["long_num", "lat_num", "phase_num", "index"])
ModeID.__doc__ = """Class for storing extrinsic mode information.

Attributes
----------
long_num : int or array_like
    The index of the longitude bin.
lat_num : int or array_like
    The index of the latitude bin.
phase_num : int or array_like
    The index of the phase bin.
index : int or array_like
    The overall mode index. Will be between 0 and 16.
"""


class LISAExtrinsicSymmetry(Reparameterisation):
    """Reparameterisation that folds the LISA extrinsic parameter space.

    This reparameterisations is based on the degeneracies described in [1] and
    the folding method proposed in [2].

    [1]: https://arxiv.org/abs/2003.00357
    [2]: https://arxiv.org/abs/2306.16429

    Parameters
    ----------
    parameters :
        List of parameter names.
    prior_bounds :
        Dictionary containing the priors bounds for each parameter.
    include_mode_index :
        If True, a discrete :code:`mode_index` parameter will be included in
        X-prime space. This makes the reparameterisations one-to-one but the
        discrete parameter must be handled appropriately when sampling.
        If False, the mode index is not stored and a random index is chosen
        when mapping from X-prime to X.
    estimate_mode_weights :
        If True, the mode weights will be estimated from the samples. This
        will only work if `include_mode_index` is False.
    minimum_mode_weight :
        Minimum weight for each mode. If the estimated weight is below this
        value, the weight will be set to this value. Set to zero or None to
        disable.
    lambda_parameter :
        Optional name for the lambda (ecliptic longitude) parameter. If not
        specified, the name will be inferred from a list of known parameters.
    beta_parameter :
        Optional name for the beta (ecliptic latitude) parameter. If not
        specified, the name will be inferred from a list of known parameters.
    psi_parameter :
        Optional name for the psi (polarization) parameter. If not
        specified, the name will be inferred from a list of known parameters.
    iota_parameter :
        Optional name for the iota (inclination) parameter. If not
        specified, the name will be inferred from a list of known parameters.
    phase_parameter :
        Optional name for the phase parameters. If not specified, the name
        will be inferred from the list of known parameters. If no such
        parameter is found, the phase will not be included.
    rng :
        Random number generator.
    """

    requires_bounded_prior = True
    one_to_one = False
    lambda_bins = (np.pi / 2) * np.arange(5)
    beta_bins = np.array([-np.pi / 2, 0.0, np.pi / 2])
    phase_bins = np.pi * np.arange(3)

    _lambda_parameter = None
    _beta_parameter = None
    _psi_parameter = None
    _iota_parameter = None
    _phase_parameter = None

    known_lambda_parameters = frozenset(
        [
            "eclipticlongitude",
            "lambda",
        ]
    )
    known_beta_parameters = frozenset(
        [
            "eclipticlatitude",
            "beta",
        ]
    )
    known_psi_parameters = frozenset(
        [
            "polarization",
            "psi",
        ]
    )
    known_iota_parameters = frozenset(
        [
            "iota",
            "inclination",
        ]
    )

    known_phase_parameters = frozenset(
        [
            "phase",
            "coa_phase",
        ]
    )

    def __init__(
        self,
        parameters: list[str] = None,
        prior_bounds: dict[str, Iterable] = None,
        include_mode_index: bool = False,
        estimate_mode_weights: bool = False,
        minimum_mode_weight: float = None,
        lambda_parameter: str = None,
        beta_parameter: str = None,
        psi_parameter: str = None,
        iota_parameter: str = None,
        phase_parameter: str = None,
        rng: np.random.Generator = None,
    ) -> None:
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng
        )

        self.lambda_parameter = lambda_parameter
        self.beta_parameter = beta_parameter
        self.psi_parameter = psi_parameter
        self.iota_parameter = iota_parameter
        self.phase_parameter = phase_parameter
        self.include_mode_index = include_mode_index

        self.prime_parameters = [p + "_folded" for p in self.parameters]

        self.estimate_mode_weights = estimate_mode_weights
        self.minimum_mode_weight = minimum_mode_weight
        self.mode_weights = None

        if self.estimate_mode_weights and self.include_mode_index:
            raise RuntimeError(
                "Cannot estimate mode weights with `mode_index=True`"
            )

        if self.include_mode_index:
            self.prime_parameters.append("mode_index")

    def determine_parameter(
        self, known_parameters: frozenset, required: bool = True
    ) -> str:
        """Determine the parameter name from a set of known parameters.

        Parameters
        ----------
        known_parameters : frozenset
            Set of known parameter names for the given parameter.
        required : bool
            If True, an error will be raised if no parameters match. If False,
            None will be returned if no parameters match.
        """
        params = set(self.parameters)
        names = list(known_parameters.intersection(params))
        if len(names) > 1:
            raise RuntimeError("Multiple parameters match")
        elif not names:
            if not required:
                return None
            raise RuntimeError("No parameters match")
        else:
            name = names[0]
        return name

    @property
    def lambda_parameter(self) -> str:
        return self._lambda_parameter

    @lambda_parameter.setter
    def lambda_parameter(self, name: Union[str, None]) -> None:
        if name is None:
            name = self.determine_parameter(self.known_lambda_parameters)
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        self._lambda_parameter = name

    @property
    def beta_parameter(self) -> str:
        return self._beta_parameter

    @beta_parameter.setter
    def beta_parameter(self, name: Union[str, None]) -> None:
        if name is None:
            name = self.determine_parameter(self.known_beta_parameters)
        if not np.isclose(self.prior_bounds[name][0], -np.pi / 2):
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], np.pi / 2):
            raise RuntimeError
        self._beta_parameter = name

    @property
    def psi_parameter(self) -> str:
        return self._psi_parameter

    @psi_parameter.setter
    def psi_parameter(self, name: Union[str, None]) -> None:
        if name is None:
            name = self.determine_parameter(self.known_psi_parameters)
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], np.pi):
            raise RuntimeError
        self._psi_parameter = name

    @property
    def iota_parameter(self):
        return self._iota_parameter

    @iota_parameter.setter
    def iota_parameter(self, name: Union[str, None]):
        if name is None:
            name = self.determine_parameter(self.known_iota_parameters)
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], np.pi):
            raise RuntimeError
        self._iota_parameter = name

    @property
    def phase_parameter(self):
        return self._phase_parameter

    @phase_parameter.setter
    def phase_parameter(self, name: Union[str, None]):
        if name is None:
            name = self.determine_parameter(
                self.known_phase_parameters, required=False
            )
            if name is None:
                return
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        self._phase_parameter = name

    @property
    def lambda_parameter_prime(self):
        return self._lambda_parameter + "_folded"

    @property
    def beta_parameter_prime(self):
        return self._beta_parameter + "_folded"

    @property
    def psi_parameter_prime(self):
        return self._psi_parameter + "_folded"

    @property
    def iota_parameter_prime(self):
        return self._iota_parameter + "_folded"

    @property
    def phase_parameter_prime(self):
        return self._phase_parameter + "_folded"

    @property
    def n_modes(self):
        """Number of modes

        Depending on the value of `phase_parameter`, the number of modes will
        be either 8 or 16.
        """
        if self.phase_parameter:
            return 16
        else:
            return 8

    def update(self, x):
        """Update the reparameterisation state."""
        if self.estimate_mode_weights:
            mode_ids = self.determine_modes(x)
            counts = np.bincount(mode_ids.index, minlength=self.n_modes)
            mode_weights = counts / counts.sum()
            if self.minimum_mode_weight:
                mode_weights = np.maximum(
                    mode_weights, self.minimum_mode_weight
                )
            self.mode_weights = mode_weights / mode_weights.sum()
        return x

    def reset(self) -> None:
        """Reset the reparameterisation."""
        self.mode_weights = None

    def determine_modes(self, x: np.ndarray) -> ModeID:
        """Determine the mode indices for each sample."""
        long_num = np.digitize(x[self.lambda_parameter], self.lambda_bins) - 1
        lat_num = np.digitize(x[self.beta_parameter], self.beta_bins) - 1
        if self.phase_parameter:
            phase_num = (
                np.digitize(x[self.phase_parameter], self.phase_bins) - 1
            )
        else:
            phase_num = 0
        index = (long_num + (lat_num * 4)) + (8 * phase_num)
        return ModeID(long_num, lat_num, phase_num, index)

    def unfold_modes(self, mode_index: np.ndarray) -> ModeID:
        """Unfold the mode index into the mode parameters."""
        # Will be zero if mode_index < 8
        phase_num = mode_index // 8
        long_num = (mode_index - (8 * phase_num)) % 4
        lat_num = (mode_index - (8 * phase_num)) // 4
        return ModeID(long_num, lat_num, phase_num, mode_index)

    def sample_mode_index(self, size: int) -> np.ndarray:
        """Sample a mode index.

        If `estimate_mode_weights` is True, the mode index will be sampled
        according to the estimated mode weights. Otherwise, the mode index will
        be sampled uniformly.

        If `estimate_mode_weights` is True, `update` must be called before
        calling this method.
        """
        if self.estimate_mode_weights:
            return self.rng.choice(
                self.n_modes, size=size, p=self.mode_weights
            )
        else:
            return self.rng.choice(self.n_modes, size=size)

    def fold(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode_ids = self.determine_modes(x)

        if self.include_mode_index:
            x_prime["mode_index"] = mode_ids.index
        x_prime[self.psi_parameter_prime] = np.mod(
            x[self.psi_parameter] - (mode_ids.long_num * 0.5 * np.pi),
            np.pi,
        )
        x_prime[self.psi_parameter_prime] = np.where(
            mode_ids.lat_num,
            x_prime[self.psi_parameter_prime],
            np.pi - x_prime[self.psi_parameter_prime],
        )
        x_prime[self.iota_parameter_prime] = np.where(
            mode_ids.lat_num,
            x[self.iota_parameter],
            np.pi - x[self.iota_parameter],
        )
        x_prime[self.beta_parameter_prime] = np.where(
            mode_ids.lat_num,
            x[self.beta_parameter],
            -x[self.beta_parameter],
        )
        x_prime[self.lambda_parameter_prime] = np.mod(
            x[self.lambda_parameter] - mode_ids.long_num * 0.5 * np.pi,
            2 * np.pi,
        )
        if self.phase_parameter:
            x_prime[self.phase_parameter_prime] = np.mod(
                x[self.phase_parameter],
                np.pi,
            )
        return x, x_prime, log_j

    def unfold(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.include_mode_index:
            mode_index = x_prime["mode_index"].copy()
        else:
            mode_index = self.sample_mode_index(size=x.shape[0])

        mode_ids = self.unfold_modes(mode_index)

        x[self.lambda_parameter] = np.mod(
            x_prime[self.lambda_parameter_prime]
            + mode_ids.long_num * 0.5 * np.pi,
            2 * np.pi,
        )
        x[self.beta_parameter] = np.where(
            mode_ids.lat_num,
            x_prime[self.beta_parameter_prime],
            -x_prime[self.beta_parameter_prime],
        )
        x[self.iota_parameter] = np.where(
            mode_ids.lat_num,
            x_prime[self.iota_parameter_prime],
            np.pi - x_prime[self.iota_parameter_prime],
        )
        x[self.psi_parameter] = np.where(
            mode_ids.lat_num,
            x_prime[self.psi_parameter_prime],
            np.pi - x_prime[self.psi_parameter_prime],
        )
        x[self.psi_parameter] = np.mod(
            x[self.psi_parameter] + (mode_ids.long_num * 0.5 * np.pi),
            np.pi,
        )
        if self.phase_parameter:
            x[self.phase_parameter] = (
                x_prime[self.phase_parameter_prime]
                + mode_ids.phase_num * np.pi
            )
        return x, x_prime, log_j

    def reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, x_prime, log_j = self.fold(x, x_prime, log_j)
        return x, x_prime, log_j

    def inverse_reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, x_prime, log_j = self.unfold(x, x_prime, log_j)
        return x, x_prime, log_j
