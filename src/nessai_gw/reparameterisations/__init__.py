from nessai.reparameterisations import (
    AnglePair,
    RescaleToBounds,
    ReparameterisationDict,
    default_reparameterisations as base_reparameterisations,
)

from .distance import DistanceReparameterisation
from .phase import DeltaPhaseReparameterisation

known_reparameterisations = ReparameterisationDict()
known_reparameterisations.add_reparameterisation(
    "distance",
    DistanceReparameterisation,
    {
        "boundary_inversion": True,
        "detect_edges": True,
        "inversion_type": "duplicate",
    },
)
known_reparameterisations.add_reparameterisation(
    "time",
    RescaleToBounds,
    {"offset": True, "update_bounds": True},
)
known_reparameterisations.add_reparameterisation(
    "sky-ra-dec",
    AnglePair,
    {"convention": "ra-dec"},
)
known_reparameterisations.add_reparameterisation(
    "sky-az-zen",
    AnglePair,
    {"convention": "az-zen"},
)
known_reparameterisations.add_reparameterisation(
    "mass_ratio",
    RescaleToBounds,
    {
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "duplicate",
        "update_bounds": True,
    },
)
known_reparameterisations.add_reparameterisation(
    "mass",
    RescaleToBounds,
    {"update_bounds": True},
)
known_reparameterisations.add_reparameterisation(
    "delta_phase",
    DeltaPhaseReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "delta-phase",
    DeltaPhaseReparameterisation,
    {},
)

known_reparameterisations.update(base_reparameterisations)
