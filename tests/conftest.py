from typing import Callable

import numpy as np
import pytest

_RNG = np.random.default_rng(1234)


@pytest.fixture()
def rng():
    return _RNG


@pytest.fixture(params=[1, 100])
def n_samples(request):
    return request.param


@pytest.fixture()
def injection_parameters():
    injection_parameters = dict(
        mass_ratio=0.9,
        chirp_mass=25.0,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=2000.0,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )
    return injection_parameters


@pytest.fixture()
def get_bilby_priors_and_likelihood():
    def get_priors_and_likelihood(parameters, injection_parameters):
        import bilby

        priors = bilby.gw.prior.BBHPriorDict()
        fixed_params = [
            "chirp_mass",
            "mass_ratio",
            "phi_12",
            "phi_jl",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "ra",
            "dec",
            "luminosity_distance",
            "geocent_time",
            "theta_jn",
            "psi",
            "phase",
        ]
        try:
            fixed_params.remove(parameters)
        except ValueError:
            for p in parameters:
                fixed_params.remove(p)
        priors["geocent_time"] = bilby.core.prior.Uniform(
            minimum=injection_parameters["geocent_time"] - 0.1,
            maximum=injection_parameters["geocent_time"] + 0.1,
            name="geocent_time",
            latex_label="$t_c$",
            unit="$s$",
        )
        for key in fixed_params:
            if key in injection_parameters:
                priors[key] = injection_parameters[key]

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=4,
            sampling_frequency=256,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,  # noqa
        )

        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=["H1"],
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization="phase" not in fixed_params,
            distance_marginalization=False,
            time_marginalization=False,
            reference_frame="sky",
        )

        likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        return priors, likelihood

    return get_priors_and_likelihood


@pytest.fixture()
def get_bilby_gw_model(get_bilby_priors_and_likelihood) -> Callable:
    """Return a function will provide a nessai model given parameters
    and an injection.
    """
    from nessai_bilby.model import BilbyModel

    def get_model(parameters, injection_parameters) -> BilbyModel:
        priors, likelihood = get_bilby_priors_and_likelihood(
            parameters, injection_parameters
        )
        return BilbyModel(priors=priors, likelihood=likelihood)

    return get_model
