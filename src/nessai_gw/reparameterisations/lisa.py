"""Reparameterisations for LISA"""

from collections import namedtuple
from typing import Iterable, Literal, Union

import numpy as np
from nessai.livepoint import empty_structured_array
from nessai.reparameterisations.base import Reparameterisation

from .utils import determine_parameter_name

SkyModeID = namedtuple("SkyModeID", ["long_num", "lat_num", "skymodeID"])
SkyModeID.__doc__ = """Class for storing sky-mode information.

Attributes
----------
long_num : int or array_like
    The index of the longitude bin.
lat_num : int or array_like
    The index of the latitude bin.
skymodeID : int or array_like
    The overall sky mode index. Its range depends on the selected symmetry
    group.
"""


class LISAExtrinsicSymmetry(Reparameterisation):
    """Reparameterisation that folds the LISA extrinsic parameter space.

    This reparameterisations is based on the degeneracies described in [1] and
    the folding method proposed in [2].

    [1]: https://arxiv.org/abs/2003.00357
    [2]: https://arxiv.org/abs/2306.16429

    Notes
    -----

    The reparameterisation is not one-to-one, as there are multiple modes in the
    original parameter space that map to the same point in the folded space. The
    `include_mode_index` option can be used to make the reparameterisation
    one-to-one, but this requires handling a discrete parameter when sampling.

    The reparameterisations can handle both ecliptic latitude and sine of
    ecliptic latitude parameters, but not both at the same time.

    Parameters
    ----------
    parameters :
        List of parameter names.
    prior_bounds :
        Dictionary containing the priors bounds for each parameter.
    include_mode_index :
        If True, a discrete :code:`skymodeID` parameter will be included in
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
    symmetry_group :
        Symmetry group used for the sky fold. Supported values are:
        ``"8-mode"`` for the full eight-mode frozen low-frequency
        symmetry including the quarter-turn longitude/polarization shifts of
        Eq. (64) in [1], and ``"reflected-antipodal"`` for the four-mode group generated
        by the reflected and antipodal transformations of Eqs. (63) and (65)
        in [1].
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
    n_phase_folds :
        Number of additional phase folds to apply after the sky fold. Must be
        1, 2, or 4 when enabled. When greater than one, a discrete
        :code:`phasemodeID` parameter is included in X-prime space. If False,
        the phase parameter is copied through unchanged.
    n_polarization_folds :
        Number of additional polarization folds to apply after the sky fold.
        Must be 1 or 2 when enabled. When greater than one, a discrete
        :code:`polarizationmodeID` parameter is included in X-prime space.
    roll :
        If True, roll the folded periodic parameters so their angular means are
        centred within the folded domains. The shifts are estimated from the
        current samples in :meth:`update`.
    rng :
        Random number generator.
    """

    requires_bounded_prior = True
    one_to_one = False
    skymode_parameter = "skymodeID"
    phase_mode_parameter = "phasemodeID"
    polarization_mode_parameter = "polarizationmodeID"
    beta_bins = np.array([-np.pi / 2, 0.0, np.pi / 2])
    sin_beta_bins = np.array([-1.0, 0.0, 1.0])
    _lambda_parameter = None
    _beta_parameter = None
    _psi_parameter = None
    _iota_parameter = None
    _phase_parameter = None
    _beta_bins = None
    _iota_mode = None
    _phase_fold = False

    symmetry_group_configs = {
        "8-mode": {"n_longitude_modes": 4, "lambda_step": np.pi / 2},
        "reflected-antipodal": {"n_longitude_modes": 2, "lambda_step": np.pi},
    }

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
            "sin_beta",
            "sinbeta",
            "sin_eclipticlatitude",
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
            "inc",
            "inclination",
            "cos_iota",
            "cosiota",
            "cos_inclination",
            "cosinc",
            "cos_inc",
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
        symmetry_group: Literal["8-mode", "reflected-antipodal"] = (
            "reflected-antipodal"
        ),
        lambda_parameter: str | None = None,
        beta_parameter: str | None = None,
        psi_parameter: str | None = None,
        iota_parameter: str | None = None,
        phase_parameter: str | None = None,
        n_phase_folds: bool | Literal[1, 2, 4] | None = False,
        n_polarization_folds: bool | Literal[1, 2] | None = False,
        roll: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng
        )

        self.lambda_parameter = lambda_parameter
        self.beta_parameter = beta_parameter
        self.psi_parameter = psi_parameter
        self.iota_parameter = iota_parameter
        self.phase_parameter = phase_parameter

        if n_phase_folds and self.phase_parameter is None:
            raise RuntimeError(
                "Phase folding is enabled but no phase parameter found"
            )
        if n_phase_folds and n_phase_folds not in (1, 2, 4):
            raise RuntimeError("n_phase_folds must be 1, 2, or 4")
        if n_polarization_folds and n_polarization_folds not in (1, 2):
            raise RuntimeError("n_polarization_folds must be 1 or 2")

        self.n_phase_folds = n_phase_folds
        self.n_polarization_folds = n_polarization_folds
        self.roll_mean = roll
        self.lambda_shift = 0.0
        self.psi_shift = 0.0
        self.phase_shift = 0.0

        self.include_mode_index = include_mode_index

        self.prime_parameters = [p + "_folded" for p in self.parameters]

        self.estimate_mode_weights = estimate_mode_weights
        self.minimum_mode_weight = minimum_mode_weight
        self.symmetry_group = self._configure_symmetry_group(symmetry_group)
        self.mode_weights = None

        if self.estimate_mode_weights and self.include_mode_index:
            raise RuntimeError(
                "Cannot estimate mode weights with `include_mode_index=True`"
            )

        if self.include_mode_index:
            self.prime_parameters.append(self.skymode_parameter)
        if self.n_phase_folds and self.n_phase_folds > 1:
            self.prime_parameters.append(self.phase_mode_parameter)
        if self.n_polarization_folds and self.n_polarization_folds > 1:
            self.prime_parameters.append(self.polarization_mode_parameter)
        self.one_to_one = self.include_mode_index

    @property
    def lambda_parameter(self) -> str:
        return self._lambda_parameter

    @lambda_parameter.setter
    def lambda_parameter(self, name: Union[str, None]) -> None:
        if name is None:
            name = determine_parameter_name(
                self.parameters, self.known_lambda_parameters
            )
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
            name = determine_parameter_name(
                self.parameters, self.known_beta_parameters
            )

        lower, upper = self.prior_bounds[name]
        if np.isclose(lower, -np.pi / 2) and np.isclose(upper, np.pi / 2):
            self._beta_bins = self.beta_bins
        elif np.isclose(lower, -1.0) and np.isclose(upper, 1.0):
            self._beta_bins = self.sin_beta_bins
        else:
            raise RuntimeError

        self._beta_parameter = name

    @property
    def psi_parameter(self) -> str:
        return self._psi_parameter

    @psi_parameter.setter
    def psi_parameter(self, name: Union[str, None]) -> None:
        if name is None:
            name = determine_parameter_name(
                self.parameters, self.known_psi_parameters
            )
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
            name = determine_parameter_name(
                self.parameters, self.known_iota_parameters
            )

        lower, upper = self.prior_bounds[name]
        if np.isclose(lower, 0.0) and np.isclose(upper, np.pi):
            self._iota_mode = "angle"
        elif np.isclose(lower, -1.0) and np.isclose(upper, 1.0):
            self._iota_mode = "cos"
        else:
            raise RuntimeError

        self._iota_parameter = name

    @property
    def phase_parameter(self):
        return self._phase_parameter

    @phase_parameter.setter
    def phase_parameter(self, name: Union[str, None]):
        if name is None:
            name = determine_parameter_name(
                self.parameters,
                self.known_phase_parameters,
                required=False,
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
    def phase_span(self) -> float:
        if not self.n_phase_folds or self.n_phase_folds == 1:
            return 2 * np.pi
        return 2 * np.pi / self.n_phase_folds

    @property
    def polarization_span(self) -> float:
        if not self.n_polarization_folds or self.n_polarization_folds == 1:
            return np.pi
        return np.pi / self.n_polarization_folds

    @property
    def lambda_step(self) -> float:
        return self.symmetry_group_configs[self.symmetry_group]["lambda_step"]

    @property
    def n_longitude_modes(self) -> int:
        return self.symmetry_group_configs[self.symmetry_group][
            "n_longitude_modes"
        ]

    @property
    def max_n_modes(self):
        """Maximum number of modes

        Number of sky modes in the selected symmetry group.
        """
        return 2 * self.n_longitude_modes

    @property
    def n_modes(self):
        """Number of modes in the selected symmetry group."""
        return self.max_n_modes

    def _configure_symmetry_group(self, symmetry_group: str) -> str:
        if symmetry_group not in self.symmetry_group_configs:
            valid = ", ".join(sorted(self.symmetry_group_configs))
            raise RuntimeError(
                f"Unknown symmetry_group: {symmetry_group}. "
                f"Valid options: {valid}"
            )
        return symmetry_group

    def update(self, x):
        """Update the reparameterisation state."""
        if self.estimate_mode_weights:
            mode_ids = self.determine_modes(x)
            counts = np.bincount(
                mode_ids.skymodeID, minlength=self.max_n_modes
            )
            mode_weights = counts / counts.sum()
            if self.minimum_mode_weight:
                mode_weights = np.maximum(
                    mode_weights, self.minimum_mode_weight
                )
            self.mode_weights = mode_weights / mode_weights.sum()
        if self.roll_mean:
            x_prime = empty_structured_array(
                x.shape[0], names=self.prime_parameters
            )
            _, x_prime, _ = self.fold(x, x_prime, np.zeros(x.shape[0]))
            self.lambda_shift = self.calculate_shift(
                x_prime[self.lambda_parameter_prime], 0.0, self.lambda_step
            )
            self.psi_shift = self.calculate_shift(
                x_prime[self.psi_parameter_prime], 0.0, self.polarization_span
            )
            if self.phase_parameter is not None:
                self.phase_shift = self.calculate_shift(
                    x_prime[self.phase_parameter_prime], 0.0, self.phase_span
                )
        return x

    def reset(self) -> None:
        """Reset the reparameterisation."""
        self.mode_weights = None

    def determine_modes(self, x: np.ndarray) -> SkyModeID:
        """Determine the mode indices for each sample."""
        long_num = np.floor(
            np.mod(x[self.lambda_parameter], 2 * np.pi) / self.lambda_step
        ).astype(int)
        lat_num = (x[self.beta_parameter] >= 0.0).astype(int)
        skymodeID = long_num + (lat_num * self.n_longitude_modes)
        return SkyModeID(long_num, lat_num, skymodeID)

    def unfold_modes(self, mode_index: np.ndarray) -> SkyModeID:
        """Unfold the mode index into the mode parameters."""
        long_num = mode_index % self.n_longitude_modes
        lat_num = mode_index // self.n_longitude_modes
        return SkyModeID(long_num, lat_num, mode_index)

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
                self.max_n_modes, size=size, p=self.mode_weights
            )
        else:
            return self.rng.choice(self.max_n_modes, size=size)

    def sample_phase_mode_index(self, size: int) -> np.ndarray:
        return self.rng.choice(self.n_phase_folds, size=size)

    def sample_polarization_mode_index(self, size: int) -> np.ndarray:
        return self.rng.choice(self.n_polarization_folds, size=size)

    def calculate_shift(
        self, x: np.ndarray, lower: float, upper: float
    ) -> float:
        """Calculate the shift to apply to x so that its mean is centred within
        the domain defined by lower and upper.

        Uses the circular mean of the angles corresponding to x.
        """
        angles = 2 * np.pi * (x - lower) / (upper - lower)
        mean_complex = np.mean(np.exp(1j * angles))
        mean_angle = np.angle(mean_complex)
        return (lower + upper) / 2 - mean_angle * (upper - lower) / (2 * np.pi)

    def _psi_longitude_shift(self, long_num: np.ndarray) -> np.ndarray:
        if self.symmetry_group == "8-mode":
            return long_num * 0.5 * np.pi
        if self.symmetry_group == "reflected-antipodal":
            return np.zeros_like(long_num, dtype=float)
        raise RuntimeError(f"Unknown symmetry group: {self.symmetry_group}")

    def fold_phase_polarization(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase_mode_index = np.floor(
            np.mod(x[self.phase_parameter], 2 * np.pi) / self.phase_span
        ).astype(int)
        polarization_mode_index = np.floor(
            np.mod(x_prime[self.psi_parameter_prime], np.pi)
            / self.polarization_span
        ).astype(int)
        x_prime[self.phase_parameter_prime] = np.mod(
            x[self.phase_parameter],
            self.phase_span,
        )
        # We use prime for polarization since it has already been folded by the
        # sky symmetries
        x_prime[self.psi_parameter_prime] = np.mod(
            x_prime[self.psi_parameter_prime],
            self.polarization_span,
        )
        if self.n_phase_folds > 1:
            x_prime[self.phase_mode_parameter] = phase_mode_index
        if self.n_polarization_folds > 1:
            x_prime[self.polarization_mode_parameter] = polarization_mode_index
        return x, x_prime, log_j

    def roll(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roll the folded parameters so their means are centred within the
        folded domains.
        """
        x_prime[self.lambda_parameter_prime] = np.mod(
            x_prime[self.lambda_parameter_prime] + self.lambda_shift,
            self.lambda_step,
        )
        x_prime[self.psi_parameter_prime] = np.mod(
            x_prime[self.psi_parameter_prime] + self.psi_shift,
            self.polarization_span,
        )
        if self.phase_parameter is not None:
            x_prime[self.phase_parameter_prime] = np.mod(
                x_prime[self.phase_parameter_prime] + self.phase_shift,
                self.phase_span,
            )
        return x, x_prime, log_j

    def unroll(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_prime[self.lambda_parameter_prime] = np.mod(
            x_prime[self.lambda_parameter_prime] - self.lambda_shift,
            self.lambda_step,
        )
        x_prime[self.psi_parameter_prime] = np.mod(
            x_prime[self.psi_parameter_prime] - self.psi_shift,
            self.polarization_span,
        )
        if self.phase_parameter is not None:
            x_prime[self.phase_parameter_prime] = np.mod(
                x_prime[self.phase_parameter_prime] - self.phase_shift,
                self.phase_span,
            )
        return x, x_prime, log_j

    def unfold_phase_polarization(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.n_phase_folds > 1
            and self.phase_mode_parameter in x_prime.dtype.names
        ):
            phase_mode_index = x_prime[self.phase_mode_parameter].astype(int)
        else:
            phase_mode_index = self.sample_phase_mode_index(size=x.shape[0])
        if (
            self.n_polarization_folds > 1
            and self.polarization_mode_parameter in x_prime.dtype.names
        ):
            polarization_mode_index = x_prime[
                self.polarization_mode_parameter
            ].astype(int)
        else:
            polarization_mode_index = self.sample_polarization_mode_index(
                size=x.shape[0]
            )

        x[self.phase_parameter] = (
            x_prime[self.phase_parameter_prime]
            + phase_mode_index * self.phase_span
        )
        x_prime[self.psi_parameter_prime] = (
            x_prime[self.psi_parameter_prime]
            + polarization_mode_index * self.polarization_span
        )
        return x, x_prime, log_j

    def _reflect_iota(self, iota: np.ndarray) -> np.ndarray:
        if self._iota_mode == "angle":
            return np.pi - iota
        elif self._iota_mode == "cos":
            return -iota
        else:
            raise RuntimeError("Unknown inclination mode")

    def fold(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode_ids = self.determine_modes(x)

        if self.include_mode_index:
            x_prime[self.skymode_parameter] = mode_ids.skymodeID
        psi_shift = self._psi_longitude_shift(mode_ids.long_num)
        x_prime[self.psi_parameter_prime] = np.mod(
            x[self.psi_parameter] - psi_shift,
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
            self._reflect_iota(x[self.iota_parameter]),
        )
        x_prime[self.beta_parameter_prime] = np.where(
            mode_ids.lat_num,
            x[self.beta_parameter],
            -x[self.beta_parameter],
        )
        x_prime[self.lambda_parameter_prime] = np.mod(
            x[self.lambda_parameter] - mode_ids.long_num * self.lambda_step,
            self.lambda_step,
        )
        if self.n_phase_folds or self.n_polarization_folds:
            x, x_prime, log_j = self.fold_phase_polarization(x, x_prime, log_j)
        elif self.phase_parameter is not None:
            x_prime[self.phase_parameter_prime] = x[self.phase_parameter]
        return x, x_prime, log_j

    def unfold(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.n_phase_folds or self.n_polarization_folds:
            x_prime_working = x_prime.copy()
            x, x_prime_working, log_j = self.unfold_phase_polarization(
                x, x_prime_working, log_j
            )
        else:
            x_prime_working = x_prime
        if self.include_mode_index:
            mode_index = x_prime_working[self.skymode_parameter].astype(int)
        else:
            mode_index = self.sample_mode_index(size=x.shape[0])

        mode_ids = self.unfold_modes(mode_index)
        psi_shift = self._psi_longitude_shift(mode_ids.long_num)

        x[self.lambda_parameter] = np.mod(
            x_prime_working[self.lambda_parameter_prime]
            + mode_ids.long_num * self.lambda_step,
            2 * np.pi,
        )
        x[self.beta_parameter] = np.where(
            mode_ids.lat_num,
            x_prime_working[self.beta_parameter_prime],
            -x_prime_working[self.beta_parameter_prime],
        )
        x[self.iota_parameter] = np.where(
            mode_ids.lat_num,
            x_prime_working[self.iota_parameter_prime],
            self._reflect_iota(x_prime_working[self.iota_parameter_prime]),
        )
        x[self.psi_parameter] = np.where(
            mode_ids.lat_num,
            x_prime_working[self.psi_parameter_prime],
            np.pi - x_prime_working[self.psi_parameter_prime],
        )
        x[self.psi_parameter] = np.mod(
            x[self.psi_parameter] + psi_shift,
            np.pi,
        )
        if not (self.n_phase_folds or self.n_polarization_folds):
            if self.phase_parameter is not None:
                x[self.phase_parameter] = x_prime_working[
                    self.phase_parameter_prime
                ]
        return x, x_prime, log_j

    def reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, x_prime, log_j = self.fold(x, x_prime, log_j)
        if self.roll_mean:
            x, x_prime, log_j = self.roll(x, x_prime, log_j)
        return x, x_prime, log_j

    def inverse_reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.roll_mean:
            x, x_prime, log_j = self.unroll(x, x_prime, log_j)
        x, x_prime, log_j = self.unfold(x, x_prime, log_j)
        return x, x_prime, log_j
