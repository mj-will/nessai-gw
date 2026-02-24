import numpy as np
import pytest
from nessai.livepoint import empty_structured_array
from nessai.utils.testing import assert_structured_arrays_equal

from nessai_gw.reparameterisations.lisa import LISAExtrinsicSymmetry


@pytest.fixture(params=["eclipticlatitude", "sin_eclipticlatitude"])
def beta_parameter_name(request):
    return request.param


@pytest.fixture
def extrinsic_parameters(beta_parameter_name):
    return [
        "eclipticlongitude",
        beta_parameter_name,
        "polarization",
        "iota",
    ]


@pytest.fixture
def extrinsic_prior_bounds(beta_parameter_name):
    beta_bounds = (
        [-np.pi / 2, np.pi / 2]
        if beta_parameter_name == "eclipticlatitude"
        else [-1.0, 1.0]
    )
    return {
        "eclipticlongitude": [0, 2 * np.pi],
        beta_parameter_name: beta_bounds,
        "polarization": [0, np.pi],
        "iota": [0, np.pi],
    }


@pytest.mark.parametrize("include_phase", [False, True])
@pytest.mark.integration_test
def test_extrinsic_reparam_invertible(
    extrinsic_parameters, extrinsic_prior_bounds, rng, n_samples, include_phase
):
    if include_phase:
        extrinsic_parameters = extrinsic_parameters + ["phase"]
        extrinsic_prior_bounds = {
            **extrinsic_prior_bounds,
            "phase": [0, 2 * np.pi],
        }

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

    x_re, _, log_j_re = reparam.inverse_reparameterise(
        x.copy(), x_prime.copy(), log_j.copy()
    )

    assert_structured_arrays_equal(x_re, x, atol=1e-14)
    np.testing.assert_equal(log_j_re, 0.0)


def test_determine_modes_expected_indices(
    extrinsic_parameters, extrinsic_prior_bounds
):
    reparam = LISAExtrinsicSymmetry(
        parameters=extrinsic_parameters,
        prior_bounds=extrinsic_prior_bounds,
    )

    n = 8
    x = empty_structured_array(n, reparam.parameters)
    x["eclipticlongitude"] = np.array(
        [0.1, np.pi / 2 + 0.1, np.pi + 0.1, 3 * np.pi / 2 + 0.1] * 2
    )
    x[reparam.beta_parameter] = np.array([0.2] * 4 + [-0.2] * 4)
    x["polarization"] = 0.5
    x["iota"] = 1.0

    mode_ids = reparam.determine_modes(x)

    np.testing.assert_array_equal(mode_ids.long_num, [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(mode_ids.lat_num, [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(mode_ids.phase_num, np.zeros(n, dtype=int))
    np.testing.assert_array_equal(mode_ids.index, [4, 5, 6, 7, 0, 1, 2, 3])


def test_unfold_modes_with_phase_indices(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
    )

    mode_index = np.arange(16)
    mode_ids = reparam.unfold_modes(mode_index)

    np.testing.assert_array_equal(mode_ids.phase_num, [0] * 8 + [1] * 8)
    np.testing.assert_array_equal(
        mode_ids.long_num,
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        mode_ids.lat_num,
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    )

    reconstructed = (
        mode_ids.long_num + 4 * mode_ids.lat_num + 8 * mode_ids.phase_num
    )
    np.testing.assert_array_equal(reconstructed, mode_index)
