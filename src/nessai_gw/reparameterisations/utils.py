from typing import Callable, Sequence, Union

from nessai.reparameterisations import (
    get_reparameterisation as get_base_reparameterisation,
)


class MissingParameterError(RuntimeError):
    """Raised when a required parameter is missing for a reparameterisation."""


class DuplicateParameterError(RuntimeError):
    """Raised when multiple parameters match for a reparameterisation."""


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


def determine_parameter_name(
    parameters: Sequence[str],
    known_parameters: frozenset[str],
    required: bool = True,
) -> str | None:
    """Determine the parameter name to use for a reparameterisation.

    Parameters
    ----------
    parameters : Sequence[str]
        List of parameter names to check.
    known_parameters : frozenset[str]
        Set of known parameter names for the reparameterisation.
    required : bool, optional
        Whether to raise an error if no parameters match. Default is True.

    Returns
    -------
    str or None
        The parameter name to use for the reparameterisation, or None if not
        required and no parameters match.

    Raises
    ------
    MissingParameterError
        If no parameters match and the parameter is required.
    DuplicateParameterError
        If multiple parameters match.
    """
    parameters = set(parameters)
    names = list(known_parameters.intersection(parameters))
    if len(names) > 1:
        raise DuplicateParameterError(
            f"Multiple parameters match! "
            f"Expected one of: {known_parameters}. "
            f"Received: {parameters}."
        )
    elif not names:
        if not required:
            return None
        raise MissingParameterError(
            "No parameters match. "
            f"Expected one of: {known_parameters}. "
            f"Received: {parameters}."
        )
    else:
        name = names[0]
    return name
