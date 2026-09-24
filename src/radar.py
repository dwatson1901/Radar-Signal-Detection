import numpy as np
from scipy import signal

#Generate a complex baseband LFM chirp
def generate_chirp(sample_rate: float, duration: float, start_frequency: float, end_frequency: float) -> np.ndarray:

    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    bandwidth = end_frequency - start_frequency
    phase = 2 * np.pi * (start_frequency * t + (bandwidth / (2 * duration)) * t**2)
    
    return np.exp(1j * phase)


#Simulate a delayed radar return with additive Gaussian noise
def simulate_target(transmitted_signal: np.ndarray, sample_rate: float, delay_seconds: float, noise_power: float = 0.1) -> np.ndarray:

    delay_samples = int(delay_seconds * sample_rate)
    delayed_signal = np.zeros(len(transmitted_signal) + delay_samples, dtype=complex,)
    delayed_signal[delay_samples: delay_samples + len(transmitted_signal)] = transmitted_signal
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(delayed_signal)) + 1j * np.random.randn(len(delayed_signal)))

    return delayed_signal + noise


#Apply a matched filter using correlation
def matched_filter(received_signal: np.ndarray, transmitted_signal: np.ndarray) -> np.ndarray:
    
    return signal.correlate(received_signal, transmitted_signal, mode="full", method="fft")