import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def get_equal_temperament_frequencies():
    return {
        "C0": 16.35, "C#0": 17.32, "D0": 18.35, "Eb0": 19.45, "E0": 20.6, "F0": 21.83, "F#0": 23.12,
        "G0": 24.5, "G#0": 25.96, "A0": 27.5, "Bb0": 29.14, "B0": 30.87,
        "C1": 32.7, "C#1": 34.65, "D1": 36.71, "Eb1": 38.89, "E1": 41.2, "F1": 43.65, "F#1": 46.25,
        "G1": 49.0, "G#1": 51.91, "A1": 55.0, "Bb1": 58.27, "B1": 61.74,
        "C2": 65.41, "C#2": 69.3, "D2": 73.42, "Eb2": 77.78, "E2": 82.41, "F2": 87.31, "F#2": 92.5,
        "G2": 98.0, "G#2": 103.83, "A2": 110.0, "Bb2": 116.54, "B2": 123.47,
        "C3": 130.81, "C#3": 138.59, "D3": 146.83, "Eb3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 185.0,
        "G3": 196.0, "G#3": 207.65, "A3": 220.0, "Bb3": 233.08, "B3": 246.94,
        "C4": 261.63, "C#4": 277.18, "D4": 293.66, "Eb4": 311.13, "E4": 329.63, "F4": 349.23, "F#4": 369.99,
        "G4": 392.0, "G#4": 415.3, "A4": 440.0, "Bb4": 466.16, "B4": 493.88,
        "C5": 523.25, "C#5": 554.37, "D5": 587.33, "Eb5": 622.25, "E5": 659.25, "F5": 698.46, "F#5": 739.99,
        "G5": 783.99, "G#5": 830.61, "A5": 880.0, "Bb5": 932.33, "B5": 987.77,
        "C6": 1046.5, "C#6": 1108.73, "D6": 1174.66, "Eb6": 1244.51, "E6": 1318.51, "F6": 1396.91,
        "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "A6": 1760.0, "Bb6": 1864.66, "B6": 1975.53,
        "C7": 2093.0, "C#7": 2217.46, "D7": 2349.32, "Eb7": 2489.02, "E7": 2637.02, "F7": 2793.83,
        "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "A7": 3520.0, "Bb7": 3729.31, "B7": 3951.07,
        "C8": 4186.01
    }

def load_data(audio_csv_path, sheet_csv_path):
    audio_data_df = pd.read_csv(audio_csv_path)
    sheet_music_df = pd.read_csv(sheet_csv_path)
    return audio_data_df, sheet_music_df

def prepare_vectors(audio_data_df, sheet_music_df, frequencies):
    audio_vectors = [(row['start_time'], round(row['average_frequency'], 4)) for _, row in audio_data_df.iterrows()]
    sheet_vectors = [(row['start_time'], frequencies[row['note']]) for _, row in sheet_music_df.iterrows()]
    return audio_vectors, sheet_vectors

def shift_and_scale_audio_vectors(audio_vectors, shift=0.0, scale=1.0):
    return [(start * scale + shift, freq) for start, freq in audio_vectors]

def map_points_onto_sheet_music(valid_points, transformed_audio_vectors):

    # Load sheet_music_csv within the function
    sheet_music_csv_path = 'exports/sheet_music_csv.csv'
    sheet_music_csv = pd.read_csv(sheet_music_csv_path)
    note_to_frequency = get_equal_temperament_frequencies()
    sheet_music_csv['sheet_frequency'] = sheet_music_csv['note'].map(note_to_frequency)

    # Convert valid_points to a DataFrame
    data = [
        {"audio_time": audio[0], "audio_frequency": audio[1], "sheet_time": sheet[0]}
        for audio, sheet in valid_points
    ]
    df = pd.DataFrame(data)

    # Left join with sheet_music_csv
    joined_df = pd.merge(
        sheet_music_csv[['start_time', 'sheet_frequency']],  # Include calculated sheet_frequency
        df,
        how='left',
        left_on='start_time',
        right_on='sheet_time'
    )

    # Drop redundant columns and rename for clarity
    joined_df = joined_df.drop(columns=['sheet_time']).rename(columns={'start_time': 'sheet_time'})

    # Truncate rows where sheet_time exceeds the maximum audio time
    max_audio_time = df['audio_time'].max()
    joined_df = joined_df[joined_df['sheet_time'] <= max_audio_time]

    # Select and format the required columns
    result_df = joined_df[['sheet_time', 'sheet_frequency', 'audio_time', 'audio_frequency']]

    # For null values, try to find a valid match in transformed_audio_vectors between the above/below audio_times
    for index, row in result_df[result_df['audio_time'].isnull()].iterrows():
        # Get the row above and below
        above = result_df.iloc[:index].loc[~result_df['audio_time'].isnull()].iloc[-1] if index > 0 else None
        below = result_df.iloc[index + 1:].loc[~result_df['audio_time'].isnull()].iloc[0] if index < len(result_df) - 1 else None

        if above is not None and below is not None:
            time_range = (above['audio_time'], below['audio_time'])

            # Enforce time order
            if time_range[0] is not None and time_range[1] is not None and time_range[0] > time_range[1]:
                continue

            # Check for matching frequency within 5% in transformed_audio_vectors
            sheet_freq = row['sheet_frequency']

            for time, freq in transformed_audio_vectors:
                if (
                        time_range[0] <= time <= time_range[1] and
                        abs(freq - sheet_freq) / sheet_freq <= 0.05
                ):
                    # Match found, update the row
                    result_df.at[index, 'audio_time'] = time
                    result_df.at[index, 'audio_frequency'] = freq
                    break

    # Handle remaining null values (only reaches this if a single note isn't played, or an edge-case start/end point)
    for i in range(1, len(result_df)):
        # Check for consecutive rows with the same sheet_frequency
        if result_df.loc[i, 'sheet_frequency'] == result_df.loc[i - 1, 'sheet_frequency']:
            # If one row has null audio_time/audio_frequency and the other does not
            if pd.isnull(result_df.loc[i, 'audio_time']) and not pd.isnull(result_df.loc[i - 1, 'audio_time']):
                result_df.loc[i, 'audio_time'] = result_df.loc[i - 1, 'audio_time']
                result_df.loc[i, 'audio_frequency'] = result_df.loc[i - 1, 'audio_frequency']
            elif not pd.isnull(result_df.loc[i, 'audio_time']) and pd.isnull(result_df.loc[i - 1, 'audio_time']):
                result_df.loc[i - 1, 'audio_time'] = result_df.loc[i, 'audio_time']
                result_df.loc[i - 1, 'audio_frequency'] = result_df.loc[i, 'audio_frequency']

    return result_df


def run_dtw(transformed_audio, trimmed_sheet):
    path = []
    audio_mapped = set()
    sheet_mapped = set()

    # Helper function to check mapping eligibility
    def is_eligible(a_time, a_value, s_time, s_value):
        freq_diff = abs(a_value - s_value) / max(a_value, s_value)
        time_diff = abs(a_time - s_time)
        return (freq_diff <= 0.052 and time_diff <= 0.1) or (freq_diff <= 0.02 and time_diff <= .5)

    # First pass: map each point in audio to the closest eligible point in sheet
    for i, (a_time, a_value) in enumerate(transformed_audio):
        best_distance = float('inf')
        best_index = None

        for j, (s_time, s_value) in enumerate(trimmed_sheet):
            if j in sheet_mapped:
                continue

            if is_eligible(a_time, a_value, s_time, s_value):
                distance = np.sqrt((a_time - s_time) ** 2 + (a_value - s_value) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = j

        if best_index is not None:
            path.append((i, best_index))
            audio_mapped.add(i)
            sheet_mapped.add(best_index)

    # Second pass: map unmatched points in sheet to the closest eligible point in audio
    for j, (s_time, s_value) in enumerate(trimmed_sheet):
        if j in sheet_mapped:
            continue

        best_distance = float('inf')
        best_index = None

        for i, (a_time, a_value) in enumerate(transformed_audio):
            if is_eligible(a_time, a_value, s_time, s_value):
                distance = np.sqrt((a_time - s_time) ** 2 + (a_value - s_value) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

        if best_index is not None:
            path.append((best_index, j))

    # Sort the path for consistency
    path.sort()
    return path


def calculate_total_distance_with_octave_correction(transformed_audio, trimmed_sheet, path):
    total_distance = 0

    for i, j in path:
        point_audio = transformed_audio[i]
        point_sheet = trimmed_sheet[j]

        current_ratio = abs(point_audio[1] / point_sheet[1])
        halved_ratio = abs((point_audio[1] / 2) / point_sheet[1])
        if abs(halved_ratio - 1) < abs(current_ratio - 1) / 20:
            point_audio = (point_audio[0], point_audio[1] / 2)

        euclidean_distance = np.sqrt((point_audio[0] - point_sheet[0]) ** 2 + (point_audio[1] - point_sheet[1]) ** 2)
        total_distance += euclidean_distance

    return total_distance


def map_valid_points_with_octave_correction(audio_vectors, sheet_vectors):
    path = run_dtw(audio_vectors, sheet_vectors)
    valid_points = []

    for audio_idx, sheet_idx in path:
        if 0 <= audio_idx < len(audio_vectors) and 0 <= sheet_idx < len(sheet_vectors):
            audio_time, audio_freq = audio_vectors[audio_idx]
            sheet_time, sheet_freq = sheet_vectors[sheet_idx]

            current_ratio = abs(audio_freq / sheet_freq)
            halved_ratio = abs((audio_freq / 2) / sheet_freq)
            if abs(halved_ratio - 1) < abs(current_ratio - 1) / 20:
                audio_freq /= 2

            valid_points.append(((audio_time, audio_freq), (sheet_time, sheet_freq)))

    return valid_points

def analyze_intonation(valid_points):
    # Create a DataFrame with calculated Cent Error
    valid_points['cent_error'] = 1200 * np.log2(
        valid_points['audio_frequency'] / valid_points['sheet_frequency'])

    # Select and rename columns for clarity
    result_df = valid_points.rename(columns={
        'sheet_time': 'time',
        'audio_frequency': 'audio_frequency',
        'sheet_frequency': 'sheet_frequency',
        'cent_error': 'cent_error'
    })[['time', 'audio_frequency', 'sheet_frequency', 'cent_error']]

    return result_df


def trim_vectors(vectors, trim_start, trim_end):
    return vectors[trim_start:len(vectors) - trim_end]

def save_results(output_csv_path, valid_points):
    mapping_df = pd.DataFrame([
        {
            "audio_time": audio[0],
            "audio_frequency": audio[1],
            "sheet_time": sheet[0],
            "sheet_frequency": sheet[1]
        }
        for audio, sheet in valid_points
    ])
    mapping_df.to_csv(output_csv_path, index=False)

def plot_results(audio_vectors, sheet_vectors, title, valid_points=None):
    plt.figure(figsize=(12, 8))

    sheet_times = [start for start, _ in sheet_vectors]
    sheet_freqs = [freq for _, freq in sheet_vectors]
    audio_times = [start for start, _ in audio_vectors]
    audio_freqs = [freq for _, freq in audio_vectors]

    plt.plot(sheet_times, sheet_freqs, label='Sheet Music Frequencies', color='cornflowerblue', marker='o')
    plt.plot(audio_times, audio_freqs, label='Audio Frequencies', color='lightcoral', marker='o')

    if valid_points:
        for (audio_time, audio_freq), (sheet_time, sheet_freq) in valid_points:
            plt.plot([audio_time, sheet_time], [audio_freq, sheet_freq], color='green', linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid()
    plt.show()


def apply_trimming(sheet_vectors, audio_vectors, audio_start_trim, audio_end_trim, sheet_start_trim, sheet_end_trim):
    trimmed_sheet = trim_vectors(sheet_vectors, sheet_start_trim, sheet_end_trim)
    trimmed_audio = trim_vectors(audio_vectors, audio_start_trim, audio_end_trim)
    return trimmed_sheet, trimmed_audio

def calculate_dynamic_ranges(audio_vectors, sheet_vectors):
    audio_duration = max([time for time, _ in audio_vectors]) - min([time for time, _ in audio_vectors])
    sheet_duration = max([time for time, _ in sheet_vectors]) - min([time for time, _ in sheet_vectors])

    scale_min = min(0.1, sheet_duration / audio_duration)
    scale_max = max(10, sheet_duration / audio_duration)
    shift_min = -2 * abs(sheet_duration - audio_duration)
    shift_max = 2 * abs(sheet_duration - audio_duration)

    return (shift_min, shift_max), (scale_min, scale_max)


def find_optimal_transformation(audio_vectors, sheet_vectors,
                                scale_range, scale_step,
                                shift_range, shift_step):
    optimal_shift = 0
    optimal_scale = 1
    best_match_count = 0
    best_distance = float('inf')

    count = 0

    for shift in np.arange(shift_range[0], shift_range[1] + shift_step, shift_step):
        for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
            transformed_audio = shift_and_scale_audio_vectors(audio_vectors, shift=shift, scale=scale)
            path = run_dtw(transformed_audio, sheet_vectors)
            total_distance = calculate_total_distance_with_octave_correction(transformed_audio, sheet_vectors, path)
            match_count = len(path)

            # Total calculations count
            count += 1

            # Prioritize higher match count, and then lower total distance
            if match_count > best_match_count or (match_count == best_match_count and total_distance < best_distance):
                print(f"Shift: {shift}, Scale: {scale}, Total Distance: {total_distance}, Matched Points: {match_count}")
                best_distance = total_distance
                best_match_count = match_count
                optimal_shift = shift
                optimal_scale = scale

    # Final Results
    return {
        "optimal_shift": optimal_shift,
        "optimal_scale": optimal_scale,
        "best_distance": best_distance,
        "best_match_count": best_match_count,
        "count": count
    }


def interpolate_notes_with_audio_json(result_df, optimal_shift, optimal_scale, audio_json_path):
    # Load audio_json
    with open(audio_json_path, 'r') as f:
        audio_json = json.load(f)

    # Convert audio_json to a DataFrame for efficient processing
    audio_data = pd.DataFrame(audio_json)

    def get_average_frequency(subset_bounds, audio_data, sheet_frequency, optimal_shift, optimal_scale):
        # Inverse-transform the subset bounds to match the original data
        original_bounds = [
            (subset_bounds[0] - optimal_shift) / optimal_scale,
            (subset_bounds[1] - optimal_shift) / optimal_scale
        ]

        # Filter the audio data based on the original bounds
        filtered = audio_data[
            (audio_data['start_time'] >= original_bounds[0]) &
            (audio_data['end_time'] <= original_bounds[1])
        ]
        valid_frequencies = []
        for freq_list in filtered['frequency']:
            if freq_list:  # Ensure frequency list is not empty
                valid_frequencies.extend([freq for freq in freq_list if abs(freq - sheet_frequency) / sheet_frequency <= 0.1])
        if valid_frequencies:
            return np.mean(valid_frequencies)
        return np.nan

    # Group by 'Audio Frequency (Hz)'
    result = []
    for freq, group in result_df.groupby(
            (result_df['audio_frequency'] != result_df['audio_frequency'].shift()).cumsum()):
        if len(group) > 1 and group['audio_frequency'].nunique() == 1:
            # Determine bounds for the group
            min_time = group['sheet_time'].min()
            if group.index[-1] + 1 in result_df.index:
                next_group_start_time = result_df.loc[group.index[-1] + 1, 'sheet_time']
                next_group_audio_time = result_df.loc[group.index[-1] + 1, 'audio_time']
            else:
                next_group_start_time = group['sheet_time'].iloc[-1] + 1
                next_group_audio_time = group['audio_time'].iloc[-1] + 1

            # Prepare next_note_start_time for each note in the group
            group = group.sort_values('sheet_time').reset_index(drop=True)
            group['next_note_start_time'] = group['sheet_time'].shift(-1).fillna(next_group_start_time)

            # Add min_time and next_group_start_time to the group
            group['min_time'] = min_time
            group['next_group_start_time'] = next_group_start_time

            # Calculate the difference between the minimum Audio Time (s) of the current group and the Audio Time (s) of the next group
            current_group_min_audio_time = group['audio_time'].min()

            audio_time_diff = next_group_audio_time - current_group_min_audio_time

            # Prepare bounds for subset mapping
            total_duration = (next_group_start_time - min_time)

            # Calculate subset_bounds column directly
            subset_bounds_list = []
            for _, row in group.iterrows():
                start_ratio = (row['sheet_time'] - min_time) / total_duration
                next_note_ratio = (row['next_note_start_time'] - min_time) / total_duration
                subset_bounds_list.append([start_ratio, next_note_ratio])
            group['subset_bounds'] = subset_bounds_list

            # Calculate search_bounds column directly
            search_bounds_list = []
            for bounds in subset_bounds_list:
                start_bound = current_group_min_audio_time + bounds[0] * audio_time_diff
                end_bound = current_group_min_audio_time + bounds[1] * audio_time_diff
                search_bounds_list.append([start_bound, end_bound])
            group['search_bounds'] = search_bounds_list

            # Calculate subset_average_frequency column directly
            subset_average_frequency_list = []
            for i, row in group.iterrows():
                search_bounds = row['search_bounds']
                sheet_frequency = row['sheet_frequency']
                subset_average_frequency = get_average_frequency(search_bounds, audio_data, sheet_frequency, optimal_shift, optimal_scale)
                subset_average_frequency_list.append(subset_average_frequency)
            group['subset_average_frequency'] = subset_average_frequency_list

            # Assign subset_average_frequency to Audio Frequency (Hz)
            group['audio_frequency'] = group['subset_average_frequency']

            result.append(group)
        else:
            # Append single entries without modification
            result.append(group)

    # Concatenate all processed groups
    updated_df = pd.concat(result, ignore_index=True).sort_values(by='sheet_time')

    # Remove additional columns created during processing
    columns_to_remove = ['next_note_start_time', 'min_time', 'next_group_start_time',
                         'subset_bounds', 'search_bounds', 'subset_average_frequency']
    updated_df = updated_df.drop(columns=columns_to_remove, errors='ignore')

    return updated_df


def map_frequency_vectors(audio_csv_path, sheet_csv_path, exports_dir="exports"):
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)

    # Load and prepare data
    frequencies = get_equal_temperament_frequencies()
    audio_data_df, sheet_music_df = load_data(audio_csv_path, sheet_csv_path)
    audio_vectors, sheet_vectors = prepare_vectors(audio_data_df, sheet_music_df, frequencies)

    # Plot un-normalized data
    plot_results(audio_vectors, sheet_vectors, title="Un-normalized Audio and Sheet Music Alignment")

    # Optimize transformation
    audio_duration = audio_vectors[-1][0] - audio_vectors[0][0]
    sheet_duration = sheet_vectors[-1][0] - sheet_vectors[0][0]

    min_scale_range = min(sheet_duration / audio_duration, 0.5)
    max_scale_range = max(sheet_duration / (audio_duration / 2), 1.5)
    shift_range = (-audio_duration, sheet_duration)

    # Rough optimization
    print("Scale range:", (min_scale_range, max_scale_range), "Step:", 0.1)
    print("Shift range:", shift_range, "Step:", 5)
    transformation_results = find_optimal_transformation(
        audio_vectors, sheet_vectors,
        scale_range=(min_scale_range, max_scale_range), scale_step=0.1,
        shift_range=shift_range, shift_step=1)

    optimal_shift = transformation_results["optimal_shift"]
    optimal_scale = transformation_results["optimal_scale"]
    best_distance = transformation_results["best_distance"]
    count = transformation_results["count"]

    print(f"Total calculations: {count}")
    print(f"Optimal Shift: {round(optimal_shift, 2)}, Optimal Scale: {round(optimal_scale, 2)}")
    print("Best Distance:", best_distance)


    # Precise optimization
    print("Scale range:", (min_scale_range, max_scale_range), "Step:", 0.1)
    print("Shift range:", shift_range, "Step:", 1)
    transformation_results = find_optimal_transformation(
        audio_vectors, sheet_vectors,
        scale_range=(optimal_scale - 0.1, optimal_scale + 0.1), scale_step=0.1,
        shift_range=(optimal_shift - 3, optimal_shift + 3), shift_step=0.25)

    optimal_shift = transformation_results["optimal_shift"]
    optimal_scale = transformation_results["optimal_scale"]
    best_distance = transformation_results["best_distance"]
    count = transformation_results["count"]

    print(f"Total calculations: {count}")
    print(f"Optimal Shift: {round(optimal_shift, 2)}, Optimal Scale: {round(optimal_scale, 2)}")
    print("Best Distance:", best_distance)

    # Apply optimized shift and scale
    transformed_audio_vectors = shift_and_scale_audio_vectors(audio_vectors, shift=optimal_shift, scale=optimal_scale)

    # Map vectors with octave correction
    valid_points = map_valid_points_with_octave_correction(transformed_audio_vectors, sheet_vectors)

    # Save original points to CSV
    save_results(os.path.join(exports_dir, 'original_points.csv'), valid_points)

    # Construct vectors with applied octave correction
    adjusted_audio_vectors = [(audio[0], audio[1]) for audio, _ in valid_points]
    adjusted_sheet_vectors = [(sheet[0], sheet[1]) for _, sheet in valid_points]

    # Plot results
    plot_results(adjusted_audio_vectors, adjusted_sheet_vectors, "Optimized Audio and Sheet Music Alignment", valid_points)

    # Map points onto sheet music
    # For each null audio point, find a matching point in transformed_audio_vectors
    # whose start time is between the start time before and after
    valid_points_df = map_points_onto_sheet_music(valid_points, transformed_audio_vectors)
    valid_points_df.to_csv(os.path.join(exports_dir, 'original_points_mapped.csv'), index=False)

    # Handle cases where no notes are mapped (like when algo doesn't detect separation between repeated notes)
    processed_points = interpolate_notes_with_audio_json(valid_points_df, optimal_shift, optimal_scale, "exports/audio_json.json")

    # Analyze intonation errors
    intonation = analyze_intonation(processed_points)

    intonation.to_csv(os.path.join(exports_dir, 'processed_intonation.csv'), index=False)

    print(f"Aggregated results saved to: {os.path.join(exports_dir, 'one_to_one_mapping_to_sheet_music.csv')}")

    print(f"Results saved to {os.path.join(exports_dir, 'original_intonation_errors.csv')}")
