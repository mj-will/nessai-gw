from nessai.reparameterisations import (
    get_reparameterisation as get_base_reparameterisation,
)
from typing import Callable, Union


def get_reparameterisation(reparameterisation: Union[str, Callable]):
    """
    Get a reparameterisation from the default list plus specific GW
    classes.

    Parameters
    ----------
    reparameterisation : str, \
            :obj:`nessai.reparameterisations.Reparameterisation`
        Name of the reparameterisations to return or a class that inherits from
        :obj:`~nessai.reparameterisations.Reparameterisation`

    Returns
    -------
    :obj:`nessai.reparameteristaions.Reparameterisation`
        Reparameterisation class.
    dict
        Keyword arguments for the specific reparameterisation.
    """
    from . import known_reparameterisations

    return get_base_reparameterisation(
        reparameterisation, defaults=known_reparameterisations
    )
