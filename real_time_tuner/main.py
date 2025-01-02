import wave
import numpy as np
import os
from r_and_d.functions import zero_padded_fourier_transform, plot_fourier, normalize_amplitude, find_all_spikes, \
    group_frequencies, find_all_fundamental_frequencies, analyze_intervals

# Path to the audio file
audio_path = "/Users/branan/Downloads/IMG_4293.wav"

# Read audio file
with wave.open(audio_path, 'rb') as wf:
    n_channels = wf.getnchannels()
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()

    raw_data = wf.readframes(n_frames)
    audio_data = np.frombuffer(raw_data, dtype=np.int16)
    audio_data = audio_data.reshape(-1, n_channels)

import json

# Define analysis parameters
start_time = 0
end_time = 200
window_duration = 0.3  # seconds
step_size = 0.1  # seconds

# Create exports directory
output_dir = "exports"
os.makedirs(output_dir, exist_ok=True)

# Initialize variables
graph_index = 0
current_end_time = 0
frequencies_data = []

# Moving window loop
while current_end_time <= end_time:
    current_start_time = max(0, current_end_time - window_duration)

    frequencies, amplitude = zero_padded_fourier_transform(
        audio_data, frame_rate, current_start_time, current_end_time, scale_n_by=10)

    if amplitude.any() and frequencies.any():
        result = find_all_spikes(frequencies, amplitude)
        groups = group_frequencies(result)
        fundamental_frequencies = find_all_fundamental_frequencies(groups)

        intonation = analyze_intervals(fundamental_frequencies)

        # Save graph with unique name
        plot_name = f"plot_{graph_index}"
        plot_fourier(frequencies, normalize_amplitude(amplitude), 0, 2500, "Frequency Distribution", plot_name)

        # Store the data
        frequencies_data.append({
            "plot": plot_name,
            "frequency": fundamental_frequencies,
            "intonation": intonation,
            "start_time": current_start_time,
            "end_time": current_end_time
        })

    # Increment the end time by the step size
    current_end_time += step_size
    graph_index += 1


# Save frequencies data to a JSON file
with open("frequencies.json", "w") as f:
    json.dump(frequencies_data, f)
