from nessai_gw.reparameterisations import known_reparameterisations
from nessai_gw.reparameterisations.distance import (
    DistanceReparameterisation,
)
from nessai_gw.reparameterisations.utils import (
    get_reparameterisation,
)
import pytest
from unittest.mock import patch


def test_get_gw_reparameterisation():
    """Test getting the gw reparameterisations.

    Assert the correct defaults are used.
    """
    expected = "out"
    with patch(
        "nessai_gw.reparameterisations.utils.get_base_reparameterisation",
        return_value=expected,
    ) as base_fn:
        out = get_reparameterisation("mass_ratio")
    assert out == expected
    base_fn.assert_called_once_with(
        "mass_ratio", defaults=known_reparameterisations
    )


@pytest.mark.integration_test
def test_get_gw_reparameterisation_integration():
    """Integration test for get_reparameterisation"""
    reparam, _ = get_reparameterisation("distance")
    assert reparam is DistanceReparameterisation
