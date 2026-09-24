import numpy as np

from src.radar import generate_chirp, simulate_target, matched_filter


def test_chirp_is_generated():
    signal = generate_chirp(
        sample_rate = 20e6,
        duration = 10e-6,
        start_frequency = 1e6,
        end_frequency = 5e6,
    )

    assert len(signal) == 200
    assert np.iscomplexobj(signal)


def test_target_delay_is_detected():

    sample_rate = 20e6
    pulse_duration = 10e-6
    target_delay = 25e-6

    transmitted = generate_chirp(
        sample_rate = sample_rate,
        duration = pulse_duration,
        start_frequency = 1e6,
        end_frequency = 5e6
    )

    received = simulate_target(
        transmitted_signal = transmitted,
        sample_rate = sample_rate,
        delay_seconds = target_delay,
        noise_power = 0.0
    )

    correlation = matched_filter(
        received_signal = received,
        transmitted_signal = transmitted
    )

    peak_index = np.argmax(np.abs(correlation))

    sample_delay = peak_index - (len(transmitted) - 1)

    detected_delay = sample_delay / sample_rate

    assert abs(detected_delay - target_delay) < 1 / sample_rate