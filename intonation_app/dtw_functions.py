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
        (row['start_time'], round(row['average_frequency'], 4), row['note_count'])
        for _, row in audio_data_df.iterrows()
    ]
    sheet_vectors = [
        (row['start_time'], frequencies[row['note']])
        for _, row in sheet_music_df.iterrows()
    ]
    return audio_vectors, sheet_vectors


def shift_and_scale_audio_vectors(audio_vectors, shift=0.0, scale=1.0):
    return [(start * scale + shift, freq, note_count) for start, freq, note_count in audio_vectors]


def map_points_onto_sheet_music(valid_points, transformed_audio_vectors):

    # Load sheet_music_csv within the function
    sheet_music_csv_path = 'exports/sheet_music_csv.csv'
    sheet_music_csv = pd.read_csv(sheet_music_csv_path)
    note_to_frequency = get_equal_temperament_frequencies()
    sheet_music_csv['sheet_frequency'] = sheet_music_csv['note'].map(note_to_frequency)

    # Convert valid_points to a DataFrame
    data = [
        {"audio_time": audio[0], "audio_frequency": audio[1], "sheet_time": sheet[0], "sheet_frequency": sheet[1]}
        for audio, sheet in valid_points
    ]
    df = pd.DataFrame(data)

    # Left join with sheet_music_csv
    joined_df = pd.merge(
        sheet_music_csv[['start_time', 'sheet_frequency']],  # Include calculated sheet_frequency
        df,
        how='left',
        left_on=['start_time', 'sheet_frequency'],
        right_on=['sheet_time', 'sheet_frequency']
    )

    # Drop redundant columns and rename for clarity
    joined_df = joined_df.drop(columns=['sheet_time']).rename(columns={'start_time': 'sheet_time'})

    # Truncate rows where sheet_time exceeds the maximum audio time
    max_audio_time = df['audio_time'].max()
    max_audio_time = joined_df.loc[joined_df['sheet_time'] > max_audio_time, 'sheet_time'].min() or max_audio_time
    joined_df = joined_df[joined_df['sheet_time'] <= max_audio_time]

    # Select and format the required columns
    result_df = joined_df[['sheet_time', 'sheet_frequency', 'audio_time', 'audio_frequency']]

    for index, row in result_df[result_df['audio_time'].isnull()].iterrows():
        # Get the row above
        above = (
            result_df.iloc[:index]
            .loc[~result_df['audio_time'].isnull()]
            .iloc[-1]
            if index > 0 and not result_df.iloc[:index].loc[~result_df['audio_time'].isnull()].empty
            else None
        )

        # Get the row below
        valid_below = result_df.iloc[index + 1:].loc[~result_df['audio_time'].isnull()]
        below = valid_below.iloc[0] if not valid_below.empty else None

        if above is not None and below is not None:
            time_range = (above['audio_time'], below['audio_time'])

            # Enforce time order
            if time_range[0] is not None and time_range[1] is not None and time_range[0] > time_range[1]:
                continue

            # Check for matching frequency within 5% in transformed_audio_vectors
            sheet_freq = row['sheet_frequency']

            for time, freq, _ in transformed_audio_vectors:
                if (
                        time_range[0] <= time <= time_range[1] and
                        abs(freq - sheet_freq) / sheet_freq <= 0.052
                ):
                    # Match found, update the row
                    result_df.at[index, 'audio_time'] = time
                    result_df.at[index, 'audio_frequency'] = freq
                    break

    return result_df


def run_dtw(transformed_audio, trimmed_sheet):
    path = []
    audio_mapped = set()
    sheet_mapped = set()

    # Helper function to check mapping eligibility
    def is_eligible(a_time, a_value, s_time, s_value):
        freq_diff = abs(a_value - s_value) / max(a_value, s_value)
        time_diff = abs(a_time - s_time)
        return (freq_diff <= 0.052 and time_diff <= 0.1) or (freq_diff <= 0.02 and time_diff <= 0.5)

    # First pass: map each point in audio to the closest eligible point in sheet
    for i, (a_time, a_value, _) in enumerate(transformed_audio):  # Unpack note_count but ignore it
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

        for i, (a_time, a_value, _) in enumerate(transformed_audio):  # Unpack note_count but ignore it
            if i in audio_mapped:
                continue

            if is_eligible(a_time, a_value, s_time, s_value):
                distance = np.sqrt((a_time - s_time) ** 2 + (a_value - s_value) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

        if best_index is not None:
            path.append((best_index, j))
            audio_mapped.add(best_index)
            sheet_mapped.add(j)

    # Sort the path for consistency
    path.sort()
    return path


def calculate_total_distance(transformed_audio, trimmed_sheet, path):
    total_distance = 0

    for i, j in path:
        point_audio = transformed_audio[i]
        point_sheet = trimmed_sheet[j]

        euclidean_distance = np.sqrt((point_audio[0] - point_sheet[0]) ** 2 + (point_audio[1] - point_sheet[1]) ** 2)
        total_distance += euclidean_distance

    return total_distance


def map_valid_points(audio_vectors, sheet_vectors):
    path = run_dtw(audio_vectors, sheet_vectors)
    valid_points = []

    for audio_idx, sheet_idx in path:
        if 0 <= audio_idx < len(audio_vectors) and 0 <= sheet_idx < len(sheet_vectors):
            audio_time, audio_freq, _ = audio_vectors[audio_idx]
            sheet_time, sheet_freq = sheet_vectors[sheet_idx]

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

    # Extract times and frequencies for sheet vectors
    sheet_times = [start for start, _ in sheet_vectors]
    sheet_freqs = [freq for _, freq in sheet_vectors]

    # Extract times and frequencies for audio vectors dynamically
    try:
        # Assuming the presence of `note_count`
        audio_times = [start for start, _, _ in audio_vectors]
        audio_freqs = [freq for _, freq, _ in audio_vectors]
    except ValueError:
        # Fallback for vectors without `note_count`
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

    def calculate_weighted_match_count(path, transformed_audio):
        return sum(10 + transformed_audio[i][2] for i, _ in path)  # Sum of note_count from matched audio points

    for shift in np.arange(shift_range[0], shift_range[1] + shift_step, shift_step):
        for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
            transformed_audio = shift_and_scale_audio_vectors(audio_vectors, shift=shift, scale=scale)
            path = run_dtw(transformed_audio, sheet_vectors)
            total_distance = calculate_total_distance(transformed_audio, sheet_vectors, path)
            weighted_match_count = calculate_weighted_match_count(path, transformed_audio)

            # Total calculations count
            count += 1

            # Prioritize higher weighted match count, and then lower total distance
            if weighted_match_count > best_match_count or (weighted_match_count == best_match_count and total_distance < best_distance):
                print(f"Shift: {shift}, Scale: {scale}, Total Distance: {total_distance}, Weighted Matched Points: {weighted_match_count}")
                best_distance = total_distance
                best_match_count = weighted_match_count
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


import json
import pandas as pd
import numpy as np

import json
import pandas as pd
import numpy as np

import json
import pandas as pd
import numpy as np

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