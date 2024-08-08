import pytest
from nessai_gw.reparameterisations import (
    DeltaPhaseReparameterisation,
)
from nessai.livepoint import (
    dict_to_live_points,
    empty_structured_array,
)
from nessai.utils.testing import assert_structured_arrays_equal
import numpy as np
from unittest.mock import patch, create_autospec


@pytest.fixture
def delta_phase_reparam():
    return create_autospec(DeltaPhaseReparameterisation)


def test_delta_phase_init(delta_phase_reparam):
    """Assert the parent method is called and the parameters are set."""
    parameters = "phase"
    prior_bounds = {"phase": [0, 6.28]}
    with patch(
        "nessai_gw.reparameterisations.phase.Reparameterisation.__init__"
    ) as mock:
        DeltaPhaseReparameterisation.__init__(
            delta_phase_reparam,
            parameters=parameters,
            prior_bounds=prior_bounds,
        )
    mock.assert_called_once_with(
        parameters=parameters, prior_bounds=prior_bounds
    )
    assert delta_phase_reparam.requires == ["psi", "theta_jn"]
    assert delta_phase_reparam.prime_parameters == ["delta_phase"]


def test_delta_phase_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=1.0, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=np.nan, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_out == x
    assert x_prime_out["delta_phase"] == 1.5
    assert log_j_out == 0


def test_delta_phase_inverse_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=np.nan, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=0.5, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.inverse_reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_prime_out == x_prime
    assert x["phase"] == 0.0
    assert log_j_out == 0


@pytest.mark.integration_test
def test_delta_phase_inverse_invertible():
    """Assert the reparameterisation is invertible"""
    n = 10
    parameters = ["phase"]
    prior_bounds = {"phase": [0.0, 2 * np.pi]}
    reparam = DeltaPhaseReparameterisation(
        parameters=parameters, prior_bounds=prior_bounds
    )
    x = dict_to_live_points(
        {
            "phase": np.random.uniform(0, 2 * np.pi, n),
            "psi": np.random.uniform(0, np.pi, n),
            "theta_jn": np.random.uniform(0, np.pi, n),
        }
    )
    x_prime = empty_structured_array(
        n, names=["delta_phase", "theta_jn", "psi"]
    )
    x_prime["psi"] = x["psi"]
    x_prime["theta_jn"] = x["theta_jn"]
    log_j = np.zeros(n)
    x_f, x_prime_f, log_j_f = reparam.reparameterise(x, x_prime, log_j)

    x_in = x_f.copy()
    x_in["phase"] = np.nan

    assert_structured_arrays_equal(x_f, x)
    np.testing.assert_array_equal(log_j_f, log_j)
    x_i, x_prime_i, log_j_i = reparam.inverse_reparameterise(
        x_in, x_prime_f.copy(), log_j_f.copy()
    )
    assert_structured_arrays_equal(x_prime_i, x_prime_f)
    assert_structured_arrays_equal(x_i, x, rtol=1e-10)
    np.testing.assert_array_equal(log_j_i, log_j_f)
