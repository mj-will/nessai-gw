import numpy as np
from nessai.reparameterisations.base import Reparameterisation
from typing import Iterable, Union


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
    """

    requires_bounded_prior = True
    one_to_one = False
    lambda_bins = (np.pi / 2) * np.arange(5)
    beta_bins = np.array([-np.pi / 2, 0.0, np.pi / 2])

    _lambda_parameter = None
    _beta_parameter = None
    _psi_parameter = None
    _iota_parameter = None

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

    def __init__(
        self,
        parameters: list[str] = None,
        prior_bounds: dict[str, Iterable] = None,
        include_mode_index: bool = False,
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
        self.include_mode_index = include_mode_index

        self.prime_parameters = [p + "_folded" for p in self.parameters]

        if self.include_mode_index:
            self.prime_parameters.append("mode_index")

    def determine_parameter(self, known_parameters: frozenset) -> str:
        params = set(self.parameters)
        names = list(known_parameters.intersection(params))
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

    def reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        long_num = np.digitize(x[self.lambda_parameter], self.lambda_bins) - 1
        lat_num = np.digitize(x[self.beta_parameter], self.beta_bins) - 1

        if self.include_mode_index:
            x_prime["mode_index"] = long_num + (lat_num * 4)
        x_prime[self.psi_parameter_prime] = np.mod(
            x[self.psi_parameter] - (long_num * 0.5 * np.pi),
            np.pi,
        )
        x_prime[self.psi_parameter_prime] = np.where(
            lat_num,
            x_prime[self.psi_parameter_prime],
            np.pi - x_prime[self.psi_parameter_prime],
        )
        x_prime[self.iota_parameter_prime] = np.where(
            lat_num,
            x[self.iota_parameter],
            np.pi - x[self.iota_parameter],
        )
        x_prime[self.beta_parameter_prime] = np.where(
            lat_num,
            x[self.beta_parameter],
            -x[self.beta_parameter],
        )
        x_prime[self.lambda_parameter_prime] = np.mod(
            x[self.lambda_parameter] - long_num * 0.5 * np.pi, 2 * np.pi
        )
        return x, x_prime, log_j

    def inverse_reparameterise(
        self, x: np.ndarray, x_prime: np.ndarray, log_j: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.include_mode_index:
            mode_index = x_prime["mode_index"]
        else:
            mode_index = np.random.choice(8, size=x.size)
        long_num = mode_index % 4
        lat_num = mode_index // 4

        x[self.lambda_parameter] = np.mod(
            x_prime[self.lambda_parameter_prime] + long_num * 0.5 * np.pi, 2 * np.pi
        )
        x[self.beta_parameter] = np.where(
            lat_num,
            x_prime[self.beta_parameter_prime],
            -x_prime[self.beta_parameter_prime],
        )
        x[self.iota_parameter] = np.where(
            lat_num,
            x_prime[self.iota_parameter_prime],
            np.pi - x_prime[self.iota_parameter_prime],
        )
        x[self.psi_parameter] = np.where(
            lat_num,
            x_prime[self.psi_parameter_prime],
            np.pi - x_prime[self.psi_parameter_prime],
        )
        x[self.psi_parameter] = np.mod(
            x[self.psi_parameter] + (long_num * 0.5 * np.pi),
            np.pi,
        )
        return x, x_prime, log_j