import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def get_equal_temperament_frequencies():
    return {
        "C0": 16.35, "C#0": 17.32, "D-0": 17.32, "D0": 18.35, "D#0": 19.45, "E-0": 19.45, "E0": 20.6,
        "F0": 21.83, "F#0": 23.12, "G-0": 23.12, "G0": 24.5, "G#0": 25.96, "A-0": 25.96, "A0": 27.5,
        "A#0": 29.14, "B-0": 29.14, "B0": 30.87,
        "C1": 32.7, "C#1": 34.65, "D-1": 34.65, "D1": 36.71, "D#1": 38.89, "E-1": 38.89, "E1": 41.2,
        "F1": 43.65, "F#1": 46.25, "G-1": 46.25, "G1": 49.0, "G#1": 51.91, "A-1": 51.91, "A1": 55.0,
        "A#1": 58.27, "B-1": 58.27, "B1": 61.74,
        "C2": 65.41, "C#2": 69.3, "D-2": 69.3, "D2": 73.42, "D#2": 77.78, "E-2": 77.78, "E2": 82.41,
        "F2": 87.31, "F#2": 92.5, "G-2": 92.5, "G2": 98.0, "G#2": 103.83, "A-2": 103.83, "A2": 110.0,
        "A#2": 116.54, "B-2": 116.54, "B2": 123.47,
        "C3": 130.81, "C#3": 138.59, "D-3": 138.59, "D3": 146.83, "D#3": 155.56, "E-3": 155.56,
        "E3": 164.81, "F3": 174.61, "F#3": 185.0, "G-3": 185.0, "G3": 196.0, "G#3": 207.65,
        "A-3": 207.65, "A3": 220.0, "A#3": 233.08, "B-3": 233.08, "B3": 246.94,
        "C4": 261.63, "C#4": 277.18, "D-4": 277.18, "D4": 293.66, "D#4": 311.13, "E-4": 311.13,
        "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G-4": 369.99, "G4": 392.0, "G#4": 415.3,
        "A-4": 415.3, "A4": 440.0, "A#4": 466.16, "B-4": 466.16, "B4": 493.88,
        "C5": 523.25, "C#5": 554.37, "D-5": 554.37, "D5": 587.33, "D#5": 622.25, "E-5": 622.25,
        "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G-5": 739.99, "G5": 783.99, "G#5": 830.61,
        "A-5": 830.61, "A5": 880.0, "A#5": 932.33, "B-5": 932.33, "B5": 987.77,
        "C6": 1046.5, "C#6": 1108.73, "D-6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E-6": 1244.51,
        "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "G-6": 1479.98, "G6": 1567.98, "G#6": 1661.22,
        "A-6": 1661.22, "A6": 1760.0, "A#6": 1864.66, "B-6": 1864.66, "B6": 1975.53,
        "C7": 2093.0, "C#7": 2217.46, "D-7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "E-7": 2489.02,
        "E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "G-7": 2959.96, "G7": 3135.96, "G#7": 3322.44,
        "A-7": 3322.44, "A7": 3520.0, "A#7": 3729.31, "B-7": 3729.31, "B7": 3951.07,
        "C8": 4186.01
    }

def load_data(audio_csv_path, sheet_csv_path):
    audio_data_df = pd.read_csv(audio_csv_path)
    sheet_music_df = pd.read_csv(sheet_csv_path)
    return audio_data_df, sheet_music_df

def prepare_vectors(audio_data_df, sheet_music_df, frequencies):
    audio_vectors = [
        (row['start_time'], round(row['frequency'], 4), row['note_count'])
        for _, row in audio_data_df.iterrows()
    ]
    sheet_vectors = [
        (row['start_time'], row['end_time'], frequencies[row['note']])
        for _, row in sheet_music_df.iterrows()
    ]
    return audio_vectors, sheet_vectors


def shift_and_scale_audio_vectors(audio_vectors, shift=0.0, scale=1.0):
    return [(start * scale + shift, freq, note_count) for start, freq, note_count in audio_vectors]


def map_points_onto_sheet_music(valid_points):
    # Load sheet_music_csv within the function
    sheet_music_csv_path = 'exports/sheet_music_csv.csv'
    sheet_music_csv = pd.read_csv(sheet_music_csv_path)
    note_to_frequency = get_equal_temperament_frequencies()
    sheet_music_csv['sheet_frequency'] = sheet_music_csv['note'].map(note_to_frequency)

    # Convert valid_points to a DataFrame
    data = [
        {"audio_time": round(audio[0], 2), "audio_frequency": audio[1],
         "sheet_start_time": sheet[0], "sheet_end_time": sheet[1],
         "sheet_frequency": sheet[2]}
        for audio, sheet in valid_points
    ]
    df = pd.DataFrame(data)

    # Join with sheet_music_csv on sheet_start_time, sheet_end_time, and sheet_frequency
    joined_df = pd.merge(
        sheet_music_csv[['start_time', 'end_time', 'sheet_frequency']],
        df,
        how='left',
        left_on=['start_time', 'end_time', 'sheet_frequency'],
        right_on=['sheet_start_time', 'sheet_end_time', 'sheet_frequency']
    )

    # Select and format the required columns
    result_df = joined_df[['audio_time', 'audio_frequency', 'sheet_start_time', 'sheet_end_time', 'sheet_frequency']]

    # Sort the DataFrame by sheet_start_time, sheet_end_time, and audio_time in ascending order
    result_df = result_df.sort_values(by=['sheet_start_time', 'sheet_end_time', 'audio_time'], ascending=True)

    return result_df


def run_dtw(transformed_audio, trimmed_sheet):
    mapping = {}

    # Helper function to check mapping eligibility
    def is_eligible(a_start_time, a_freq, s_start_time, s_end_time, s_freq):
        # Check if the audio note's start time falls within the sheet's time range
        if not (s_start_time <= a_start_time < s_end_time):
            return False
        # Check if the frequency difference is within the allowed threshold
        freq_diff = abs(a_freq - s_freq) / max(a_freq, s_freq)
        return freq_diff <= 0.045

    # Iterate over each audio note
    for a_time, a_value, _ in transformed_audio:  # Unpack note_count but ignore it
        # Iterate over each entry in the trimmed sheet
        for s_start_time, s_end_time, s_value in trimmed_sheet:
            if is_eligible(a_time, a_value, s_start_time, s_end_time, s_value):
                key = (s_start_time, s_end_time, s_value)

                # Add the audio note to the corresponding sheet entry
                if key not in mapping:
                    mapping[key] = []
                mapping[key].append((a_time, a_value))

    return mapping


def calculate_total_distance(transformed_audio, trimmed_sheet, path):
    total_distance = 0

    for i, j in path:
        point_audio = transformed_audio[i]
        point_sheet = trimmed_sheet[j]

        euclidean_distance = np.sqrt((point_audio[0] - point_sheet[0]) ** 2 + (point_audio[1] - point_sheet[1]) ** 2)
        total_distance += euclidean_distance

    return total_distance


def map_valid_points(mapping):
    valid_points = []

    for sheet_key, audio_values in mapping.items():
        sheet_start_time, sheet_end_time, sheet_freq = sheet_key

        for audio_time, audio_freq in audio_values:
            valid_points.append(((audio_time, audio_freq), (sheet_start_time, sheet_end_time, sheet_freq)))

    return valid_points

def analyze_intonation(valid_points):
    # Avoid division by zero or invalid log operations by filtering out problematic rows
    valid_points = valid_points.dropna(subset=['audio_frequency', 'sheet_frequency'])
    valid_points = valid_points[valid_points['audio_frequency'] > 0]
    valid_points = valid_points[valid_points['sheet_frequency'] > 0]

    # Calculate Cent Error
    valid_points['cent_error'] = 1200 * np.log2(
        valid_points['audio_frequency'] / valid_points['sheet_frequency']
    )

    # Group by 'sheet_start_time' and 'sheet_frequency' and calculate averages
    grouped_df = (
        valid_points
        .groupby(['sheet_start_time', 'sheet_frequency'], as_index=False)
        .agg({'audio_frequency': 'mean', 'cent_error': 'mean'})
    )

    return grouped_df


def trim_vectors(vectors, trim_start, trim_end):
    return vectors[trim_start:len(vectors) - trim_end]

def save_results(output_csv_path, valid_points):
    mapping_df = pd.DataFrame([
        {
            "audio_time": audio[0],
            "audio_frequency": audio[1],
            "sheet_start_time": sheet[0],
            "sheet_end_time": sheet[1],
            "sheet_frequency": sheet[2]
        }
        for audio, sheet in valid_points
    ])
    mapping_df.to_csv(output_csv_path, index=False)


def plot_results(audio_vectors, sheet_vectors, title, valid_points=None):
    plt.figure(figsize=(12, 8))

    # Plot straight lines for each tuple in sheet_vectors
    for i, (start_time, end_time, frequency) in enumerate(sheet_vectors):
        plt.plot([start_time, end_time], [frequency, frequency], color='cornflowerblue',
                 label='Sheet Music Frequencies' if i == 0 else "")
        # Add a vertical divider at the start and end of each interval
        plt.plot([start_time, start_time], [frequency - 10, frequency + 10], color='blue', linestyle=':', linewidth=1,
                 label='Interval Start' if i == 0 else "")
        plt.plot([end_time, end_time], [frequency - 10, frequency + 10], color='orange', linestyle=':', linewidth=1,
                 label='Interval End' if i == 0 else "")

    # Extract times and frequencies for audio vectors dynamically
    try:
        # Assuming the presence of `note_count`
        audio_times = [start for start, _, _ in audio_vectors]
        audio_freqs = [freq for _, freq, _ in audio_vectors]
    except ValueError:
        # Fallback for vectors without `note_count`
        audio_times = [start for start, _ in audio_vectors]
        audio_freqs = [freq for _, freq in audio_vectors]

    # Plot individual points for audio frequencies
    plt.scatter(audio_times, audio_freqs, label='Audio Frequencies', color='lightcoral', marker='.')

    # Plot valid points if provided
    if valid_points:
        for (audio_time, audio_freq), (sheet_start_time, sheet_end_time, sheet_freq) in valid_points:
            # Map audio_time to the nearest point on the sheet interval
            closest_sheet_time = max(min(audio_time, sheet_end_time), sheet_start_time)

            plt.plot([audio_time, closest_sheet_time], [audio_freq, sheet_freq], color='green', linestyle='--',
                     linewidth=2)

    # Add title and labels
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
    best_mapping = {}

    count = 0

    def calculate_match_count(mapping):
        return sum(len(v) for v in mapping.values())  # Count total matches from the dictionary

    for shift in np.arange(shift_range[0], shift_range[1] + shift_step, shift_step):
        for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
            transformed_audio = shift_and_scale_audio_vectors(audio_vectors, shift=shift, scale=scale)
            mapping = run_dtw(transformed_audio, sheet_vectors)
            match_count = calculate_match_count(mapping)

            # Total calculations count
            count += 1

            # Prioritize higher match count, and then lower total distance
            if match_count > best_match_count:
                print(f"Shift: {shift}, Scale: {scale}, Matched Points: {match_count}")
                best_match_count = match_count
                optimal_shift = shift
                optimal_scale = scale
                best_mapping = mapping

    # Final Results
    return {
        "optimal_shift": optimal_shift,
        "optimal_scale": optimal_scale,
        "best_mapping": best_mapping,
        "best_match_count": best_match_count,
        "count": count
    }


def interpolate_doubled_notes_with_audio_json(result_df, optimal_shift, optimal_scale, audio_json_path, audio_csv_path):
    # Load audio_json
    with open(audio_json_path, 'r') as f:
        audio_json = json.load(f)

    # Convert audio_json to a DataFrame for efficient processing
    audio_data = pd.DataFrame(audio_json)

    # Load audio_csv
    audio_csv = pd.read_csv(audio_csv_path)
    audio_csv['frequencies'] = audio_csv['frequencies'].apply(eval)  # Convert string lists to actual lists

    def get_average_frequency_from_csv(proportion_bounds, matched_row):
        """
        Calculate the average frequency within the bounds defined by proportions for the matched row.
        Uses the 'frequencies' column of the matched_row, which contains a list of frequencies.
        """
        # Extract the frequencies from the matched row
        frequencies = matched_row['frequencies']

        # Calculate the subset indices based on the proportion bounds
        total_frequencies = len(frequencies)
        start_index = int(proportion_bounds[0] * total_frequencies)
        end_index = int(proportion_bounds[1] * total_frequencies)

        # Ensure end_index is inclusive and errs on the side of oversampling
        end_index = min(end_index + 1, total_frequencies)

        # Select the subset of frequencies
        subset_frequencies = frequencies[start_index:end_index]

        # Calculate and return the average of the subset frequencies
        if subset_frequencies:
            return np.mean(subset_frequencies)
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
            else:
                next_group_start_time = group['sheet_time'].iloc[-1] + 1

            # Prepare next_note_start_time for each note in the group
            group = group.sort_values('sheet_time').reset_index(drop=True)
            group['next_note_start_time'] = group['sheet_time'].shift(-1).fillna(next_group_start_time)

            # Add min_time and next_group_start_time to each row in the group
            group['min_time'] = group['sheet_time'].apply(lambda x: min_time)
            group['next_group_start_time'] = group['sheet_time'].apply(lambda x: next_group_start_time)

            # Calculate individual bounds for each row using min_time and next_note_start_time
            group['bounds_start'] = (group['sheet_time'] - min_time) / (next_group_start_time - min_time)
            group['bounds_end'] = (group['next_note_start_time'] - min_time) / (next_group_start_time - min_time)

            # Reverse-transform using optimal_shift and optimal_scale for each row
            group['estimated_start_time'] = group['bounds_start'] * optimal_scale + optimal_shift
            group['estimated_end_time'] = group['bounds_end'] * optimal_scale + optimal_shift

            # Match to audio_csv based on estimated time
            subset_average_frequency_list = []
            for _, row in group.iterrows():
                matched_row = audio_csv[
                    (audio_csv['average_frequency'].round(2) == row['audio_frequency'].round(2))
                ]
                if not matched_row.empty:
                    # Take the closest row based on estimated_start_time
                    matched_row = matched_row.iloc[
                        np.argmin(abs(matched_row['start_time'] - row['estimated_start_time']))
                    ]

                    # Define proportion bounds for subset
                    proportion_bounds = [row['bounds_start'], row['bounds_end']]

                    # Calculate the average frequency using the proportion bounds
                    avg_frequency = get_average_frequency_from_csv(
                        proportion_bounds,
                        matched_row
                    )

                    # If avg_frequency is NaN, fall back to row's original audio_frequency
                    subset_average_frequency_list.append(
                        avg_frequency if not pd.isna(avg_frequency) else row['audio_frequency']
                    )
                else:
                    # If no match, keep the original audio_frequency
                    subset_average_frequency_list.append(row['audio_frequency'])

            # Assign calculated average frequencies to the group
            group['audio_frequency'] = subset_average_frequency_list

            result.append(group)
        else:
            # Append single entries without modification
            result.append(group)

    # Concatenate all processed groups
    updated_df = pd.concat(result, ignore_index=True).sort_values(by='sheet_time')

    # Remove additional columns created during processing
    columns_to_remove = ['next_note_start_time', 'min_time', 'next_group_start_time',
                         'bounds_start', 'bounds_end', 'estimated_start_time', 'estimated_end_time']
    updated_df = updated_df.drop(columns=columns_to_remove, errors='ignore')

    return updated_df


def parse_single_null_values_using_audio_json(df, audio_json_path, optimal_shift, optimal_scale):
    # Load the audio JSON file
    with open(audio_json_path, 'r') as f:
        audio_data = json.load(f)

    # Check for null rows
    null_rows = df[df['audio_time'].isnull() | df['audio_frequency'].isnull()]

    if null_rows.empty:
        return df  # No null rows, return as is

    # Iterate over null rows
    for index, row in null_rows.iterrows():
        sheet_frequency = row['sheet_frequency']

        # Find boundaries from the nearest non-null rows above and below
        above_rows = df.iloc[:index].dropna(subset=['audio_time', 'audio_frequency'])
        if index + 1 < len(df):
            below_rows = df.iloc[index + 1:]
            if 'audio_time' in below_rows.columns and 'audio_frequency' in below_rows.columns:
                below_rows = below_rows.dropna(subset=['audio_time', 'audio_frequency'])
            else:
                below_rows = pd.DataFrame()
        else:
            below_rows = pd.DataFrame()

        if above_rows.empty or below_rows.empty:
            # If no valid rows above or below, skip this row
            df = df.drop(index)
            continue

        above_row = above_rows.iloc[-1]
        below_row = below_rows.iloc[0]

        above_time = above_row['audio_time']
        below_time = below_row['audio_time']

        # Undo transformation for boundaries
        original_bounds = [
            (above_time - optimal_shift) / optimal_scale,
            (below_time - optimal_shift) / optimal_scale
        ]

        # Filter audio data within the boundaries
        filtered_audio_data = [
            entry for entry in audio_data if original_bounds[0] <= entry['start_time'] <= original_bounds[1]
        ]

        candidates = []
        for entry in filtered_audio_data:
            for freq in entry['frequency']:
                if abs(freq - sheet_frequency) / sheet_frequency <= 0.052:
                    candidates.append((freq, entry['start_time']))

        if not candidates:
            # Remove row if no matching frequency is found
            df = df.drop(index)
        else:
            # Average the matching frequencies and times
            matching_frequencies = [freq for freq, _ in candidates]
            matching_times = [time for _, time in candidates]

            avg_frequency = np.mean(matching_frequencies)
            avg_time = np.mean(matching_times)

            # Update the DataFrame
            df.at[index, 'audio_frequency'] = round(avg_frequency, 4)
            df.at[index, 'audio_time'] = (avg_time + optimal_shift) * optimal_scale

    return df