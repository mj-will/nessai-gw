from nessai.livepoint import empty_structured_array, numpy_array_to_live_points
from nessai.utils.testing import assert_structured_arrays_equal
from nessai_gw.reparameterisations.lisa import LISAExtrinsicSymmetry
import numpy as np
import pytest


@pytest.fixture
def extrinsic_parameters():
    return [
        "eclipticlongitude",
        "eclipticlatitude",
        "polarization",
        "iota",
    ]


@pytest.fixture
def extrinsic_prior_bounds():
    return {
        "eclipticlongitude": [0, 2 * np.pi],
        "eclipticlatitude": [-np.pi / 2, np.pi / 2],
        "polarization": [0, np.pi],
        "iota": [0, np.pi]
    }


@pytest.mark.integration_test
def test_extrinsic_reparam_invertible(
    extrinsic_parameters, extrinsic_prior_bounds, rng, n_samples
):

    reparam = LISAExtrinsicSymmetry(
        parameters=extrinsic_parameters,
        prior_bounds=extrinsic_prior_bounds,
        include_mode_index=True,
    )

    x = empty_structured_array(n_samples, reparam.parameters)
    x_prime = empty_structured_array(n_samples, reparam.prime_parameters)
    log_j = np.zeros(n_samples)
    for param, bounds in extrinsic_prior_bounds.items():
        x[param] = rng.uniform(*bounds, size=n_samples)

    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

    x_re, _, log_j_re = reparam.inverse_reparameterise(x.copy(), x_prime.copy(), log_j.copy())

    assert_structured_arrays_equal(x_re, x, atol=1e-14)
    np.testing.assert_equal(log_j_re, 0.0)
