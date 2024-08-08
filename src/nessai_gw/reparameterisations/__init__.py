from nessai.reparameterisations import (
    AnglePair,
    RescaleToBounds,
    default_reparameterisations as base_reparameterisations,
)

from .distance import DistanceReparameterisation
from .phase import DeltaPhaseReparameterisation

known_reparameterisations = {
    "distance": (
        DistanceReparameterisation,
        {
            "boundary_inversion": True,
            "detect_edges": True,
            "inversion_type": "duplicate",
        },
    ),
    "time": (RescaleToBounds, {"offset": True, "update_bounds": True}),
    "sky-ra-dec": (AnglePair, {"convention": "ra-dec"}),
    "sky-az-zen": (AnglePair, {"convention": "az-zen"}),
    "mass_ratio": (
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "duplicate",
            "update_bounds": True,
        },
    ),
    "mass": (RescaleToBounds, {"update_bounds": True}),
    "delta_phase": (DeltaPhaseReparameterisation, {}),
    "delta-phase": (DeltaPhaseReparameterisation, {}),
}

known_reparameterisations.update(base_reparameterisations)
