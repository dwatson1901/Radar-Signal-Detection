import numpy as np
import matplotlib.pyplot as plt

from radar import generate_chirp, simulate_target, matched_filter

SAMPLE_RATE = 20e6
PULSE_DURATION = 10e-6
START_FREQUENCY = 1e6
END_FREQUENCY = 5e6
TARGET_DELAY = 25e-6

def main():

    transmitted = generate_chirp(
        sample_rate=SAMPLE_RATE,
        duration = PULSE_DURATION,
        start_frequency = START_FREQUENCY,
        end_frequency = END_FREQUENCY
    )

    received = simulate_target(
        transmitted_signal = transmitted,
        sample_rate = SAMPLE_RATE,
        delay_seconds = TARGET_DELAY,
        noise_power = 0.5
    )

    correlation = matched_filter(
        received_signal = received,
        transmitted_signal = transmitted
    )

    magnitude = np.abs(correlation)

    peak_index = np.argmax(magnitude)

    sample_delay = peak_index - (len(transmitted) - 1)

    estimated_delay = sample_delay / SAMPLE_RATE

    speed_of_light = 3e8

    estimated_range = (speed_of_light * estimated_delay / 2)

    print(f"Estimated delay: {estimated_delay * 1e6:.2f} microseconds")
    print(f"Estimated target range: {estimated_range / 1000:.2f} km")

    plt.figure(figsize=(10, 6))

    plt.plot(magnitude)

    plt.axvline(
        peak_index,
        linestyle="--",
        label="Detected target"
    )

    plt.title("Radar Matched Filter Output")
    plt.xlabel("Sample")
    plt.ylabel("Correlation Magnitude")
    plt.legend()

    plt.tight_layout()

    plt.savefig("plots/matched_filter_output.png")

    plt.show()


if __name__ == "__main__":
    main()