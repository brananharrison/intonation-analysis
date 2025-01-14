import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import wave
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from r_and_d.functions import zero_padded_fourier_transform, plot_fourier, normalize_amplitude, find_all_spikes, \
    group_frequencies, find_all_fundamental_frequencies, identify_notes

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
        print(plot_name)
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
        return {
            "plot": plot_name,
            "frequency": [],
            "notes": [],
            "start_time": current_start_time,
            "end_time": current_end_time
        }



# Wrapper function to run audio_to_json in a separate thread
def run_audio_to_json(audio_path):
    audio_to_json(audio_path)


# Asynchronous function to call the audio_to_json function using ThreadPoolExecutor
async def async_audio_to_json(audio_path):
    loop = asyncio.get_event_loop()

    # Use ThreadPoolExecutor to run audio_to_json asynchronously
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, run_audio_to_json, audio_path)


# Function to process the audio and create the JSON output
def audio_to_json(audio_path):
    # Read audio file
    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw_data = wf.readframes(n_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # If multi-channel audio, average across channels
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

    # Define analysis parameters
    total_duration = n_frames / frame_rate  # Calculate total duration of audio
    window_duration = 0.1  # seconds
    step_size = 0.1  # seconds

    # Create exports directory
    output_dir = "exports"
    os.makedirs(output_dir, exist_ok=True)

    current_start_time = 0
    graph_index = 0
    tasks = []

    # Moving window loop
    while current_start_time < total_duration:
        current_end_time = min(current_start_time + window_duration, total_duration)

        # Add the task for this window
        tasks.append((audio_data, frame_rate, current_start_time, current_end_time, graph_index))

        # Increment start time and graph index
        current_start_time += step_size
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
    with open("exports/audio_json.json", "w") as f:
        json.dump(frequencies_data, f, indent=4)



# Call async_audio_to_json
async def main(audio_path):
    await async_audio_to_json(audio_path)


# Set up command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file to extract frequency data.")
    parser.add_argument('audio_path', type=str, help="Path to the audio file")

    args = parser.parse_args()

    # Run the asyncio event loop with the provided audio path
    asyncio.run(main(args.audio_path))
