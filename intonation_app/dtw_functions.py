import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import find_peaks


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
        (row['start_time'], round(row['frequency'], 4))
        for _, row in audio_data_df.iterrows()
    ]
    sheet_vectors = [
        (row['start_time'], row['end_time'], frequencies[row['note']])
        for _, row in sheet_music_df.iterrows()
    ]
    return audio_vectors, sheet_vectors


def shift_and_scale_audio_vectors(audio_vectors, shift=0.0, scale=1.0):
    return [(round(start * scale + shift, 3), round(freq, 4)) for start, freq in audio_vectors]


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
    for a_time, a_value in transformed_audio:  # Unpack note_count but ignore it
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


def optimize_subsets(transformed_audio_vectors, sheet_vectors, summary):

    def adjust_vectors(sheet_start, sheet_end, summary_start, summary_end, transformed_audio_vectors):
        # Remove points within the original summary interval
        remaining_vectors = [point for point in transformed_audio_vectors if not (summary_start <= point[0] <= summary_end)]

        # Calculate left and right bounds
        left_vectors = [point for point in remaining_vectors if point[0] < summary_start]
        right_vectors = [point for point in remaining_vectors if point[0] > summary_end]

        left_min = min(left_vectors, key=lambda x: x[0])[0] if left_vectors else summary_start
        left_max = max(left_vectors, key=lambda x: x[0])[0] if left_vectors else summary_start
        right_min = min(right_vectors, key=lambda x: x[0])[0] if right_vectors else summary_end
        right_max = max(right_vectors, key=lambda x: x[0])[0] if right_vectors else summary_end

        updated_vectors = []

        shift_amount = sheet_start - summary_start
        scale_amount = (sheet_end - sheet_start) / (summary_end - summary_start)

        for time, frequency in transformed_audio_vectors:
            if summary_start <= time <= summary_end:
                new_time = (time - summary_start) * scale_amount + sheet_start
                updated_vectors.append((round(new_time, 4), frequency))

        # Adjust the left and right vectors while keeping left_min and right_max anchored
        refined_vectors = []
        for point in remaining_vectors:
            time, frequency = point
            if time < summary_start:
                scale = (summary_start - left_min) / (left_max - left_min) if left_max != left_min else 1
                shift = left_min - left_min * scale
                new_time = time * scale + shift
            elif time > summary_end:
                scale = (right_max - sheet_end) / (right_max - right_min) if right_max != right_min else 1
                shift = right_max - right_max * scale
                new_time = time * scale + shift
            else:
                continue  # This should not happen; handled by the earlier filtering
            refined_vectors.append((round(new_time, 4), frequency))

        # Combine updated and refined vectors
        combined_vectors = updated_vectors + refined_vectors
        combined_vectors.sort(key=lambda x: x[0])  # Ensure time order

        return combined_vectors

    def generate_remaining_vectors(transformed_audio_vectors, transformed_intervals, summary_start, summary_end):
        remaining_vectors = []
        locked_vectors = []
        for point in transformed_audio_vectors:
            time, frequency = point
            in_transformed = any(start <= time <= end for start, end in transformed_intervals)
            if not in_transformed:
                remaining_vectors.append(point)
            else:
                locked_vectors.append(point)
        return remaining_vectors, locked_vectors

    # Initialize transformed intervals list
    transformed_intervals = []

    # Extract the relevant summary and sheet vectors
    summary_tuple_1 = summary[1]  # (0.616, 1.504, 329.535)
    sheet_tuple_1 = sheet_vectors[1]  # (0.0, 1.0, 329.63)

    summary_start_1, summary_end_1, _ = summary_tuple_1
    sheet_start_1, sheet_end_1, _ = sheet_tuple_1

    # First transformation
    transformed_vectors_1 = adjust_vectors(sheet_start_1, sheet_end_1, summary_start_1, summary_end_1, transformed_audio_vectors)
    transformed_intervals.append((sheet_start_1, sheet_end_1))

    # Extract the second summary and sheet intervals
    summary_tuple_2 = summary[17]  # (3.402, 4.058, 346.047)
    sheet_start_2 = 3.5  # (3.5, 4.0, 349.23)
    sheet_end_2 = 4.5  # (4.0, 4.5, 349.23)

    summary_start_2 = 3.3747411444141693
    summary_end_2 = 4.058

    # Generate remaining and locked vectors using transformed intervals
    remaining_vectors, locked_vectors = generate_remaining_vectors(
        transformed_vectors_1, transformed_intervals, summary_start_2, summary_end_2
    )

    # Second transformation using the selected remaining vectors
    transformed_vectors_2 = adjust_vectors(sheet_start_2, sheet_end_2, summary_start_2, summary_end_2, remaining_vectors)
    transformed_intervals.append((sheet_start_2, sheet_end_2))

    # Combine locked vectors with the second transformed vectors
    final_transformed_vectors = transformed_vectors_2 + locked_vectors
    final_transformed_vectors.sort(key=lambda x: x[0])  # Ensure time order

    return final_transformed_vectors


def find_gaps(transformed_audio_vectors, time_step=0.01, freq_step=2, radius=0.032):
    # Divides audio vectors into bins based on density
    # Returns (lower_freq_bound, upper_freq_bound): [list of time dividers]

    # Frequency bounds calculation
    freqs = [v[1] for v in transformed_audio_vectors]
    min_freq, max_freq = min(freqs), max(freqs)
    sampled_freqs = np.arange(min_freq - 5, max_freq + 5, freq_step)
    freq_densities = [
        sum(1 for f in freqs if abs(f - freq) <= freq_step) for freq in sampled_freqs
    ]
    peaks, _ = find_peaks(freq_densities)
    freq_gaps = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        middle_index = (start + end) // 2
        freq_gaps.append(sampled_freqs[middle_index])

    # Include start and end boundaries for frequency bins
    freq_gaps = [sampled_freqs[0]] + sorted(freq_gaps) + [sampled_freqs[-1]]

    # Time bounds, ensures all points are boxed in all 4 edges
    gap_dict = {}

    for i in range(len(freq_gaps) - 1):
        bin_start, bin_end = freq_gaps[i], freq_gaps[i + 1]
        bin_times = [v[0] for v in transformed_audio_vectors if bin_start <= v[1] < bin_end]
        if bin_times:
            min_time, max_time = min(bin_times), max(bin_times)
            time_bins = np.arange(min_time - radius, max_time + radius, time_step)
            time_densities = [
                sum(1 for t in bin_times if abs(t - time) <= radius) for time in time_bins
            ]

            time_gaps = []
            i = 0
            blacklist = set()
            while i < len(time_densities):
                if i in blacklist:
                    i += 1
                    continue

                if time_densities[i] == 0:
                    start = i
                    while i < len(time_densities) and time_densities[i] == 0:
                        i += 1
                    end = i - 1
                    middle_index = (start + end) // 2
                    middle_time = time_bins[middle_index]
                    time_gaps.append(middle_time)

                    # Blacklist the radius around the identified zero-density interval
                    blacklist.update(range(max(0, start - int(radius / time_step)),
                                           min(len(time_densities), end + int(radius / time_step) + 1)))
                else:
                    i += 1

            # Ensure explicit start and end time bounds for the bin
            if time_bins[0] not in time_gaps:
                time_gaps.insert(0, time_bins[0])
            if time_bins[-1] not in time_gaps:
                time_gaps.append(time_bins[-1])

            # Add the time gaps to the dictionary with the frequency bounds as key
            gap_dict[(bin_start, bin_end)] = time_gaps

    return gap_dict


def summarize_gaps(gap_dict, transformed_audio_vectors):
    # Converts (lower_freq_bound, upper_freq_bound): [list of time dividers]
    #   to list of (min_time, max_time, avg_frequency) tuples within each rectangle

    summaries = []

    for (freq_start, freq_end), time_gaps in gap_dict.items():
        # Filter the vectors within the frequency range
        for i in range(len(time_gaps) - 1):
            time_start, time_end = time_gaps[i], time_gaps[i + 1]
            vectors_in_box = [
                v for v in transformed_audio_vectors
                if freq_start <= v[1] < freq_end and time_start <= v[0] < time_end
            ]

            if vectors_in_box:
                min_time = round(float(min(v[0] for v in vectors_in_box)), 4)
                max_time = round(float(max(v[0] for v in vectors_in_box)), 4)
                avg_frequency = round(float(sum(v[1] for v in vectors_in_box) / len(vectors_in_box)), 4)
                summaries.append((min_time, max_time, avg_frequency))

    return sorted(summaries, key=lambda x: (x[0], x[1]))


def plot_results(audio_vectors, sheet_vectors, title, valid_points=None, gaps=None):
    plt.figure(figsize=(14, 8))

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

    # Plot valid points connections
    if valid_points:
        for i, ((audio_time, audio_freq), (sheet_start_time, sheet_end_time, sheet_freq)) in enumerate(valid_points):
            closest_sheet_time = max(min(audio_time, sheet_end_time), sheet_start_time)
            plt.plot([audio_time, closest_sheet_time], [audio_freq, sheet_freq], color='green', linestyle='--',
                     linewidth=2, label='Valid Connection' if i == 0 else "")

    # Plot gaps if provided
    freq_gap_legend_added = False
    if gaps:
        for (freq_lower, freq_upper), time_gaps in gaps.items():
            plt.axhline(y=freq_lower, color='purple', linestyle='--', linewidth=1,
                        label='Frequency Gaps' if not freq_gap_legend_added else "")
            plt.axhline(y=freq_upper, color='purple', linestyle='--', linewidth=1)
            freq_gap_legend_added = True

            for gap_time in time_gaps:
                plt.plot([gap_time, gap_time], [freq_lower, freq_upper], color='red', linestyle='--', linewidth=1,
                         label='Time Gap' if freq_gap_legend_added and gap_time == time_gaps[0] else "")

    # Plot configuration
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Adjust legend and layout
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', framealpha=0.8)
    plt.subplots_adjust(right=0.8)  # Leave space for the legend

    plt.grid()
    plt.show()


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

