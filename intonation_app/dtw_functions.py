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
    unmapped_points = []

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
        mapped = False
        # Iterate over each entry in the trimmed sheet
        for s_start_time, s_end_time, s_value in trimmed_sheet:
            if is_eligible(a_time, a_value, s_start_time, s_end_time, s_value):
                key = (s_start_time, s_end_time, s_value)

                # Add the audio note to the corresponding sheet entry
                if key not in mapping:
                    mapping[key] = []
                mapping[key].append((a_time, a_value))
                mapped = True

        if not mapped:
            unmapped_points.append((a_time, a_value))

    return mapping, unmapped_points


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


def optimize_subsets(transformed_audio_vectors, sheet_vectors):
    segment_count = 6
    refined_audio_vectors = []
    combined_best_mapping = {}
    combined_unmapped_points = []

    # Get the sorted times from transformed_audio_vectors
    times = sorted(t[0] for t in transformed_audio_vectors)

    # Calculate gaps between consecutive times
    gaps = [(times[i+1] - times[i], times[i], times[i+1]) for i in range(len(times) - 1)]
    # Sort gaps by size in descending order
    largest_gaps = sorted(gaps, key=lambda x: x[0], reverse=True)[:segment_count]

    # Find the midpoints of the largest gaps
    segment_boundaries = sorted([(gap[1] + gap[2]) / 2 for gap in largest_gaps])

    # Add the minimum and maximum times as boundaries
    segment_boundaries = [min(times)] + segment_boundaries + [max(times)]

    # Define overlapping sheet segment boundaries
    sheet_segment_boundaries = [
        segment_boundaries[0] - (segment_boundaries[1] - segment_boundaries[0]) / 2
    ] + [
        (segment_boundaries[i] + segment_boundaries[i+1]) / 2
        for i in range(len(segment_boundaries) - 1)
    ] + [
        segment_boundaries[-1] + (segment_boundaries[-1] - segment_boundaries[-2]) / 2
    ]
    print(sheet_segment_boundaries)

    for i in range(len(segment_boundaries) - 1):
        # Audio segment boundaries
        segment_start = segment_boundaries[i]
        segment_end = segment_boundaries[i + 1]

        # Sheet segment boundaries with overlap
        sheet_segment_start = sheet_segment_boundaries[i]
        sheet_segment_end = sheet_segment_boundaries[i + 2]

        # Extract segments for audio and sheet vectors
        segment = [
            vector for vector in transformed_audio_vectors
            if segment_start <= vector[0] < segment_end
        ]
        sheet_segment = [
            vector for vector in sheet_vectors
            if sheet_segment_start <= vector[0] < sheet_segment_end
        ]

        if not segment or not sheet_segment:
            continue  # Skip empty segments

        # Run find_optimal_transformation with reset scale and shift
        transformation_results = find_optimal_transformation(
            segment,
            sheet_segment,
            scale_range=(0.5, 1.5),  # Reset scale range around 0
            scale_step=0.01,
            shift_range=(-0.2, 0.2),  # Reset shift range around 0
            shift_step=0.025
        )

        # Extract results
        segment_optimal_shift = transformation_results["optimal_shift"]
        segment_optimal_scale = transformation_results["optimal_scale"]
        segment_best_mapping = transformation_results["best_mapping"]
        segment_unmapped_points = transformation_results["unmapped_points"]
        print(segment_optimal_shift, segment_optimal_scale)

        # Apply the optimized transformation
        refined_segment = shift_and_scale_audio_vectors(
            segment,
            shift=segment_optimal_shift,
            scale=segment_optimal_scale
        )

        # Append results
        refined_audio_vectors.extend(refined_segment)
        combined_unmapped_points.extend(segment_unmapped_points)

        # Combine the best mapping
        for key, value in segment_best_mapping.items():
            if key not in combined_best_mapping:
                combined_best_mapping[key] = value
            else:
                combined_best_mapping[key].extend(value)

    # Ensure all mappings are combined properly
    for key in combined_best_mapping:
        combined_best_mapping[key] = list(set(combined_best_mapping[key]))

    return refined_audio_vectors, combined_best_mapping, combined_unmapped_points


def find_largest_time_gaps(transformed_audio_vectors):
    # Sort the audio vectors by time for time gap calculations
    transformed_audio_vectors.sort(key=lambda x: x[0])
    time_gaps = []

    # Calculate time gaps (sorted by time)
    for i in range(len(transformed_audio_vectors) - 1):
        start_time = transformed_audio_vectors[i][0]
        end_time = transformed_audio_vectors[i + 1][0]
        time_gap_size = end_time - start_time
        time_midpoint = (start_time + end_time) / 2
        time_gaps.append((time_gap_size, time_midpoint, start_time, end_time))

    # Sort the audio vectors by frequency for frequency gap calculations
    transformed_audio_vectors.sort(key=lambda x: x[1])
    freq_gaps = []

    # Calculate frequency gaps (sorted by frequency)
    for i in range(len(transformed_audio_vectors) - 1):
        start_freq = transformed_audio_vectors[i][1]
        end_freq = transformed_audio_vectors[i + 1][1]
        freq_gap_size = abs(end_freq - start_freq)
        freq_midpoint = (start_freq + end_freq) / 2
        freq_gaps.append((freq_gap_size, freq_midpoint, start_freq, end_freq))

    # Sort the gaps in descending order by size
    time_gaps.sort(key=lambda x: x[0], reverse=True)
    freq_gaps.sort(key=lambda x: x[0], reverse=True)

    # Process time gaps
    largest_time_midpoints = []
    blacklisted_time_intervals = []

    for gap in time_gaps:
        gap_size, midpoint, start_time, end_time = gap
        if gap_size > 0.02 and all(
            end_time <= start or start_time >= end
            for start, end in blacklisted_time_intervals
        ):
            largest_time_midpoints.append(round(midpoint, 2))
            blacklisted_time_intervals.append((start_time, end_time))

    # Process frequency gaps
    largest_freq_midpoints = []
    blacklisted_freq_intervals = []

    for gap in freq_gaps:
        gap_size, midpoint, start_freq, end_freq = gap
        if gap_size > 0.035 * start_freq and all(
            end_freq <= start or start_freq >= end
            for start, end in blacklisted_freq_intervals
        ):
            largest_freq_midpoints.append(round(midpoint, 2))
            blacklisted_freq_intervals.append((min(start_freq, end_freq), max(start_freq, end_freq)))

    # Print the results
    print("Time Gaps:")
    for gap in largest_time_midpoints:
        print(f"Time Midpoint: {gap}")

    print("Frequency Gaps:")
    for gap in largest_freq_midpoints:
        print(f"Frequency Midpoint: {gap}")

    return sorted(largest_time_midpoints), sorted(largest_freq_midpoints)


def plot_results(audio_vectors, sheet_vectors, title, valid_points=None, time_gaps=None, freq_gaps=None):
    plt.figure(figsize=(12, 8))

    for i, (start_time, end_time, frequency) in enumerate(sheet_vectors):
        plt.plot([start_time, end_time], [frequency, frequency], color='cornflowerblue',
                 label='Sheet Music Frequencies' if i == 0 else "")
        plt.plot([start_time, start_time], [frequency - 10, frequency + 10], color='blue', linestyle=':', linewidth=1,
                 label='Interval Start' if i == 0 else "")
        plt.plot([end_time, end_time], [frequency - 10, frequency + 10], color='orange', linestyle=':', linewidth=1,
                 label='Interval End' if i == 0 else "")

    try:
        audio_times = [start for start, _, _ in audio_vectors]
        audio_freqs = [freq for _, freq, _ in audio_vectors]
    except ValueError:
        audio_times = [start for start, _ in audio_vectors]
        audio_freqs = [freq for _, freq in audio_vectors]

    plt.scatter(audio_times, audio_freqs, label='Audio Frequencies', color='lightcoral', marker='.')

    if valid_points:
        for (audio_time, audio_freq), (sheet_start_time, sheet_end_time, sheet_freq) in valid_points:
            closest_sheet_time = max(min(audio_time, sheet_end_time), sheet_start_time)
            plt.plot([audio_time, closest_sheet_time], [audio_freq, sheet_freq], color='green', linestyle='--',
                     linewidth=2)

    if time_gaps:
        for gap_time in time_gaps:
            plt.axvline(x=gap_time, color='black', linestyle=':', linewidth=1,
                        label='Time Gap' if gap_time == time_gaps[0] else "")

    if freq_gaps:
        freq_gap_legend_added = False
        for gap_freq in freq_gaps:
            plt.axhline(y=gap_freq, color='purple', linestyle='--', linewidth=1,
                        label='Freq Gap' if not freq_gap_legend_added else "")
            freq_gap_legend_added = True

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid()
    plt.show()


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
    best_unmapped_points = []

    count = 0

    def calculate_match_count(mapping):
        return sum(len(v) for v in mapping.values()) + sum(5 for _ in mapping)

    for shift in np.arange(shift_range[0], shift_range[1] + shift_step, shift_step):
        for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
            transformed_audio = shift_and_scale_audio_vectors(audio_vectors, shift=shift, scale=scale)
            mapping, unmapped_points = run_dtw(transformed_audio, sheet_vectors)
            match_count = calculate_match_count(mapping)

            # Total calculations count
            count += 1

            # Prioritize higher match count, and then lower total distance
            if match_count > best_match_count:
                best_match_count = match_count
                optimal_shift = shift
                optimal_scale = scale
                best_mapping = mapping
                best_unmapped_points = unmapped_points

    # Final Results
    return {
        "optimal_shift": optimal_shift,
        "optimal_scale": optimal_scale,
        "best_mapping": best_mapping,
        "best_match_count": best_match_count,
        "unmapped_points": best_unmapped_points,
        "count": count
    }

