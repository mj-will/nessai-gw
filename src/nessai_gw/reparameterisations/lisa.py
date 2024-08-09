import numpy as np
from nessai.reparameterisations.base import Reparameterisation
from typing import Iterable, Union


class LISAExtrinsicSymmetry(Reparameterisation):

    requires_bounded_prior = True
    one_to_one = False
    lambda_bins = (np.pi / 2) * np.arange(4)
    beta_bins = np.array([-np.pi / 2, 0.0, np.pi / 2])

    _lambda_parameter = None
    _beta_parameter = None
    _psi_parameter = None
    _iota_parameter = None

    known_lambda_parameters = frozenset(
        [
            "eclipticlongitude",
        ]
    )
    known_beta_parameters = frozenset(
        [
            "eclipticlatitude",
        ]
    )
    known_psi_parameters = frozenset(
        [
            "polarization",
        ]
    )
    known_iota_parameters = frozenset(
        [
            "iota",
            "inclination",
        ]
    )

    def __init__(
        self,
        parameters: list[str] = None,
        prior_bounds: dict[str, Iterable] = None,
        lambda_parameter: str = None,
        beta_parameter: str = None,
        psi_parameter: str = None,
        iota_parameter: str = None,
    ) -> None:
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.lambda_parameter = lambda_parameter
        self.beta_parameter = beta_parameter
        self.psi_parameter = psi_parameter
        self.iota_parameter = iota_parameter

    def determine_parameter(self, known_parameters: frozenset) -> str:
        params = set(self.parameters)
        names = known_parameters.intersection(params)
        if len(names) > 1:
            raise RuntimeError("Multiple parameters match")
        elif not names:
            raise RuntimeError("No parameters match")
        else:
            name = names[0]
        return name

    @property
    def lambda_parameter(self) -> str:
        return self._lambda_parameter

    @lambda_parameter.setter
    def lambda_parameter(self, name: Union[str, None]) -> None:
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_lambda_parameters)
        self._lambda_parameter = name

    @property
    def beta_parameter(self) -> str:
        return self._beta_parameter

    @beta_parameter.setter
    def beta_parameter(self, name: Union[str, None]) -> None:
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_beta_parameters)
        self._beta_parameter = name

    @property
    def psi_parameter(self) -> str:
        return self._psi_parameter

    @psi_parameter.setter
    def psi_parameter(self, name: Union[str, None]) -> None:
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_psi_parameters)
        self._psi_parameter = name

    @property
    def iota_parameter(self):
        return self._iota_parameter

    @iota_parameter.setter
    def iota_parameter(self, name: Union[str, None]):
        if self.prior_bounds[name][0] != 0:
            raise RuntimeError
        if not np.isclose(self.prior_bounds[name][1], 2 * np.pi):
            raise RuntimeError
        if name is None:
            name = self.determine_parameter(self.known_iota_parameters)
        self._iota_parameter = name

    def reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        long_num = np.digitize(x[self.lambda_parameter], self.lambda_bins) - 1
        lat_num = np.digitize(x[self.beta_parameter], self.beta_bins) - 1
        x_prime[self.psi_parameter] = np.mod(
            x[self.psi_parameter] - (long_num * 0.5 * np.pi),
            np.pi,
        )
        x_prime[self.psi_parameter] = np.where(
            lat_num,
            x_prime[self.psi_parameter],
            np.pi - x_prime[self.psi_parameter],
        )
        x_prime[self.iota_parameter] = np.where(
            lat_num,
            x[self.iota_parameter],
            np.pi - x[self.iota_parameter],
        )
        x_prime[self.beta_parameter] = np.where(
            lat_num,
            x[self.beta_parameter],
            -x[self.beta_parameter],
        )
        x_prime[self.lambda_parameter] = np.mod(
            x[self.lambda_parameter] - long_num * 0.5 * np.pi, 2 * np.pi
        )
        return x, x_prime, log_j

    def inverse_reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        sky_num = np.random.choice(8, size=x.size)
        long_num = sky_num % 4
        lat_num = sky_num // 4

        x[self.lambda_parameter] = np.mod(
            x_prime[self.lambda_parameter] + long_num * 0.5 * np.pi, 2 * np.pi
        )
        x[self.beta_parameter] = np.where(
            lat_num,
            -x_prime[self.beta_parameter],
            x_prime[self.beta_parameter],
        )
        x[self.iota_parameter] = np.where(
            lat_num,
            np.pi - x_prime[self.iota_parameter],
            x_prime[self.iota_parameter],
        )
        x[self.psi_parameter] = np.where(
            lat_num,
            np.pi - x_prime[self.psi_parameter],
            x_prime[self.psi_parameter],
        )
        x[self.psi_parameter] = np.mod(
            x[self.psi_parameter] + (long_num * 0.5 * np.pi),
            np.pi,
        )
        return x, x_prime, log_j
