import time
import wave
import numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from r_and_d.functions import zero_padded_fourier_transform, plot_fourier, normalize_amplitude, find_all_spikes, \
    group_frequencies, find_all_fundamental_frequencies, identify_notes

# Path to the audio file
audio_path = "/Users/branan/Downloads/How to play Twinkle Twinkle on the violin!.wav"

# Read audio file
with wave.open(audio_path, 'rb') as wf:
    n_channels = wf.getnchannels()
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()

    raw_data = wf.readframes(n_frames)
    audio_data = np.frombuffer(raw_data, dtype=np.int16)
    audio_data = audio_data.reshape(-1, n_channels)

# Define analysis parameters
start_time = 50
end_time = 95
window_duration = 0.1  # seconds
step_size = 0.1  # seconds

# Create exports directory
output_dir = "exports"
os.makedirs(output_dir, exist_ok=True)

def analyze_window(audio_data, frame_rate, current_start_time, current_end_time, graph_index):
    frequencies, amplitude = zero_padded_fourier_transform(
        audio_data, frame_rate, current_start_time, current_end_time, scale_n_by=10)

    if amplitude.any() and frequencies.any():
        result = find_all_spikes(frequencies, amplitude)
        groups = group_frequencies(result)
        fundamental_frequencies = find_all_fundamental_frequencies(groups)

        notes = identify_notes(fundamental_frequencies)

        # Save graph with unique name
        plot_name = f"plot_{graph_index}"
        plot_fourier(frequencies, normalize_amplitude(amplitude), 0, 2500, "Frequency Distribution", plot_name)

        # Return the data
        return {
            "plot": plot_name,
            "frequency": fundamental_frequencies,
            "notes": notes,
            "start_time": current_start_time,
            "end_time": current_end_time
        }
    else:
        # Even if no spikes are found, return a result with empty lists
        plot_name = f"plot_{graph_index}"
        plot_fourier(frequencies, normalize_amplitude(amplitude), 0, 2500, "Frequency Distribution", plot_name)
        return {
            "plot": plot_name,
            "frequency": [],
            "notes": [],
            "start_time": current_start_time,
            "end_time": current_end_time
        }


def main():
    current_end_time = 0
    graph_index = 0
    tasks = []

    # Moving window loop
    while current_end_time <= end_time:
        current_start_time = round(max(0, current_end_time - window_duration), 3)
        tasks.append((audio_data, frame_rate, current_start_time, current_end_time, graph_index))

        # Increment the end time by the step size
        current_end_time = round(current_end_time + step_size, 3)
        graph_index += 1

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=25) as executor:
        future_to_task = {executor.submit(analyze_window, *task): task[-1] for task in tasks}

        results = []
        for future in as_completed(future_to_task):
            task_index = future_to_task[future]
            result = future.result()
            if result:
                results.append((task_index, result))

    # Sort results by the original index
    results.sort(key=lambda x: x[0])
    frequencies_data = [result for _, result in results]

    # Save frequencies data to a JSON file
    with open("audio_json.json", "w") as f:
        json.dump(frequencies_data, f, indent=4)



if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds.")
