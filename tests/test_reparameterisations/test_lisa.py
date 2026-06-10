import numpy as np
import pytest
from nessai.livepoint import empty_structured_array
from nessai.utils.testing import assert_structured_arrays_equal

from nessai_gw.reparameterisations.lisa import LISAExtrinsicSymmetry


@pytest.fixture(params=["eclipticlatitude", "sin_eclipticlatitude"])
def beta_parameter_name(request):
    return request.param


@pytest.fixture(params=["iota", "cos_iota"])
def iota_parameter_name(request):
    return request.param


@pytest.fixture
def extrinsic_parameters(beta_parameter_name, iota_parameter_name):
    return [
        "eclipticlongitude",
        beta_parameter_name,
        "polarization",
        iota_parameter_name,
    ]


@pytest.fixture
def extrinsic_prior_bounds(beta_parameter_name, iota_parameter_name):
    beta_bounds = (
        [-np.pi / 2, np.pi / 2]
        if beta_parameter_name == "eclipticlatitude"
        else [-1.0, 1.0]
    )
    iota_bounds = [0, np.pi] if iota_parameter_name == "iota" else [-1.0, 1.0]
    return {
        "eclipticlongitude": [0, 2 * np.pi],
        beta_parameter_name: beta_bounds,
        "polarization": [0, np.pi],
        iota_parameter_name: iota_bounds,
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
        symmetry_group="8-mode",
    )

    n = 8
    x = empty_structured_array(n, reparam.parameters)
    x["eclipticlongitude"] = np.array(
        [0.1, np.pi / 2 + 0.1, np.pi + 0.1, 3 * np.pi / 2 + 0.1] * 2
    )
    x[reparam.beta_parameter] = np.array([0.2] * 4 + [-0.2] * 4)
    x["polarization"] = 0.5
    x[reparam.iota_parameter] = 0.5

    mode_ids = reparam.determine_modes(x)

    np.testing.assert_array_equal(mode_ids.long_num, [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(mode_ids.lat_num, [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(mode_ids.skymodeID, [4, 5, 6, 7, 0, 1, 2, 3])


def test_unfold_modes_with_phase_parameter_present(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        symmetry_group="8-mode",
    )

    mode_index = np.arange(8)
    mode_ids = reparam.unfold_modes(mode_index)

    np.testing.assert_array_equal(
        mode_ids.long_num,
        [0, 1, 2, 3, 0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        mode_ids.lat_num,
        [0, 0, 0, 0, 1, 1, 1, 1],
    )

    reconstructed = mode_ids.long_num + 4 * mode_ids.lat_num
    np.testing.assert_array_equal(reconstructed, mode_index)


def test_symmetry_group_reflected_antipodal_without_phase(
    extrinsic_parameters, extrinsic_prior_bounds
):
    reparam = LISAExtrinsicSymmetry(
        parameters=extrinsic_parameters,
        prior_bounds=extrinsic_prior_bounds,
        symmetry_group="reflected-antipodal",
    )

    assert reparam.max_n_modes == 4
    assert reparam.n_modes == 4
    assert np.isclose(reparam.lambda_step, np.pi)


def test_symmetry_group_reflected_antipodal_with_phase(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        symmetry_group="reflected-antipodal",
    )

    assert reparam.max_n_modes == 4
    assert reparam.n_modes == 4
    assert np.isclose(reparam.lambda_step, np.pi)


def test_phase_fold_does_not_expand_sky_modes(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        symmetry_group="reflected-antipodal",
        n_phase_folds=2,
        n_polarization_folds=2,
    )

    assert reparam.max_n_modes == 4
    assert reparam.n_modes == 4


def test_unknown_symmetry_group_raises(
    extrinsic_parameters, extrinsic_prior_bounds
):
    with pytest.raises(RuntimeError, match="Unknown symmetry_group"):
        LISAExtrinsicSymmetry(
            parameters=extrinsic_parameters,
            prior_bounds=extrinsic_prior_bounds,
            symmetry_group="not-a-group",
        )


def test_phase_parameter_can_be_present_without_phase_fold(
    extrinsic_parameters, extrinsic_prior_bounds, rng, n_samples
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        include_mode_index=True,
        symmetry_group="8-mode",
    )

    x = empty_structured_array(n_samples, reparam.parameters)
    x_prime = empty_structured_array(n_samples, reparam.prime_parameters)
    log_j = np.zeros(n_samples)
    for param, bounds in prior_bounds.items():
        x[param] = rng.uniform(*bounds, size=n_samples)

    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)
    x_re, _, log_j_re = reparam.inverse_reparameterise(
        x.copy(), x_prime.copy(), log_j.copy()
    )

    assert reparam.n_modes == 8
    np.testing.assert_allclose(x_prime["phase_folded"], x["phase"])
    assert_structured_arrays_equal(x_re, x, atol=1e-14)
    np.testing.assert_equal(log_j_re, 0.0)


def test_roll_centres_folded_periodic_parameters(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        symmetry_group="reflected-antipodal",
        phase_parameter="phase",
        n_phase_folds=2,
        n_polarization_folds=2,
        roll=True,
    )

    x = empty_structured_array(8, reparam.parameters)
    x[reparam.lambda_parameter] = 0.1
    x[reparam.beta_parameter] = 0.2
    x[reparam.psi_parameter] = 0.2
    x[reparam.iota_parameter] = 0.5
    x[reparam.phase_parameter] = 0.3

    reparam.update(x)

    x_prime = empty_structured_array(8, reparam.prime_parameters)
    _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(8))

    np.testing.assert_allclose(
        x_prime[reparam.lambda_parameter_prime], reparam.lambda_step / 2
    )
    np.testing.assert_allclose(
        x_prime[reparam.psi_parameter_prime], reparam.polarization_span / 2
    )
    np.testing.assert_allclose(
        x_prime[reparam.phase_parameter_prime], reparam.phase_span / 2
    )


def test_roll_preserves_invertibility_without_phase_polarization_folds(
    extrinsic_parameters, extrinsic_prior_bounds, rng, n_samples
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        include_mode_index=True,
        roll=True,
    )

    x = empty_structured_array(n_samples, reparam.parameters)
    x_prime = empty_structured_array(n_samples, reparam.prime_parameters)
    log_j = np.zeros(n_samples)
    for param, bounds in prior_bounds.items():
        x[param] = rng.uniform(*bounds, size=n_samples)

    reparam.update(x)
    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)
    x_re, _, log_j_re = reparam.inverse_reparameterise(
        x.copy(), x_prime.copy(), log_j.copy()
    )

    assert_structured_arrays_equal(x_re, x, atol=1e-14)
    np.testing.assert_equal(log_j_re, 0.0)


def test_phase_polarization_folding_adds_mode_index(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        n_phase_folds=2,
        n_polarization_folds=2,
    )

    assert "phasemodeID" in reparam.prime_parameters
    assert "polarizationmodeID" in reparam.prime_parameters


def test_phase_and_polarization_mode_ids_expected_values(
    extrinsic_parameters, extrinsic_prior_bounds
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        include_mode_index=True,
        n_phase_folds=2,
        n_polarization_folds=2,
    )

    x = empty_structured_array(4, reparam.parameters)
    x["eclipticlongitude"] = 0.1
    x[reparam.beta_parameter] = 0.2
    x["polarization"] = np.array([0.2, 0.2, 2.0, 2.0])
    x[reparam.iota_parameter] = 0.5
    x["phase"] = np.array([0.2, 1.8, np.pi + 0.2, np.pi + 1.8])

    x_prime = empty_structured_array(4, reparam.prime_parameters)
    log_j = np.zeros(4)
    _, x_prime, _ = reparam.reparameterise(x, x_prime, log_j)

    np.testing.assert_allclose(
        x_prime["phase_folded"],
        [0.2, 1.8, 0.2, 1.8],
    )
    np.testing.assert_allclose(
        x_prime["polarization_folded"],
        [0.2, 0.2, 0.42920367, 0.42920367],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        x_prime["phasemodeID"],
        [0, 0, 1, 1],
    )
    np.testing.assert_array_equal(
        x_prime["polarizationmodeID"],
        [0, 0, 1, 1],
    )
    np.testing.assert_array_equal(x_prime["skymodeID"], [2, 2, 2, 2])


@pytest.mark.integration_test
def test_extrinsic_reparam_invertible_with_phase_polarization_folding(
    extrinsic_parameters, extrinsic_prior_bounds, rng, n_samples
):
    parameters = extrinsic_parameters + ["phase"]
    prior_bounds = {**extrinsic_prior_bounds, "phase": [0, 2 * np.pi]}

    reparam = LISAExtrinsicSymmetry(
        parameters=parameters,
        prior_bounds=prior_bounds,
        include_mode_index=True,
        n_phase_folds=2,
        n_polarization_folds=2,
    )

    x = empty_structured_array(n_samples, reparam.parameters)
    x_prime = empty_structured_array(n_samples, reparam.prime_parameters)
    log_j = np.zeros(n_samples)
    for param, bounds in prior_bounds.items():
        x[param] = rng.uniform(*bounds, size=n_samples)

    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)
    x_re, x_prime_re, log_j_re = reparam.inverse_reparameterise(
        x.copy(), x_prime.copy(), log_j.copy()
    )

    assert_structured_arrays_equal(x_re, x, atol=1e-14)
    assert_structured_arrays_equal(x_prime_re, x_prime, atol=1e-14)
    np.testing.assert_equal(log_j_re, 0.0)


def test_sin_iota_not_supported():
    parameters = [
        "eclipticlongitude",
        "eclipticlatitude",
        "polarization",
        "sin_iota",
    ]
    prior_bounds = {
        "eclipticlongitude": [0, 2 * np.pi],
        "eclipticlatitude": [-np.pi / 2, np.pi / 2],
        "polarization": [0, np.pi],
        "sin_iota": [0.0, 1.0],
    }

    with pytest.raises(RuntimeError):
        LISAExtrinsicSymmetry(
            parameters=parameters,
            prior_bounds=prior_bounds,
        )
