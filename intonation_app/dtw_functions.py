import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict


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

def filter_audio_vectors(audio_vectors, sheet_vectors, tolerance=0.06):
    max_sheet_frequency = max(freq for _, _, freq in sheet_vectors)
    return [
        (start_time, frequency)
        for start_time, frequency in audio_vectors
        if frequency <= max_sheet_frequency * (1 + tolerance)
    ]

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
        if not (s_start_time <= a_start_time <= s_end_time):
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
                continue

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


def optimize_subsets(transformed_audio_vectors, sheet_vectors, summary, rounds=12):
    def adjust_vectors(sheet_start, sheet_end, summary_start, summary_end, summary, transformed_audio_vectors, transformed_intervals):
        updated_vectors = []

        scale_amount = (sheet_end - sheet_start) / (summary_end - summary_start)

        sheet_ends = [end for _, end in transformed_intervals]
        sheet_starts = [start for start, _ in transformed_intervals]
        audio_times = [time for time, _ in transformed_audio_vectors]

        left_min = max([end for end in sheet_ends if end < summary_start], default=min(audio_times))
        left_min = left_min if left_min != summary_start else None

        right_max = min([start for start in sheet_starts if start > summary_end], default=max(audio_times))
        right_max = right_max if right_max != summary_end else None


        # Iterate over all transformed_audio_vectors and process them
        for time, frequency in transformed_audio_vectors:
            if summary_start <= time <= summary_end:
                new_time = round((time - summary_start) * scale_amount + sheet_start, 4)

            elif left_min is not None and left_min <= time < summary_start:
                uniform_scale = (time - left_min) / (summary_start - left_min)
                shift = uniform_scale * (sheet_start - left_min)
                new_time = round(left_min + shift, 4)

            elif left_min is None or time < left_min:
                new_time = round(time, 4)

            elif right_max is not None and summary_end < time <= right_max:
                uniform_scale = (time - summary_end) / (right_max - summary_end)
                shift = uniform_scale * (right_max - sheet_end)
                new_time = round(sheet_end + shift, 4)

            elif right_max is None or time > right_max:
                new_time = round(time, 4)


            for i in range(len(summary)):
                s_start, s_end, s_freq, s_num, freq_bin = summary[i]
                if time == s_start:
                    summary[i] = (new_time, s_end, s_freq, s_num, freq_bin)
                elif time == s_end:
                    summary[i] = (s_start, new_time, s_freq, s_num, freq_bin)

            updated_vectors.append((new_time, frequency))

        updated_vectors.sort(key=lambda x: x[0])  # Ensure time order

        return updated_vectors, summary

    def find_best_transform(summary, sheet_vectors, transformed_intervals):
        # Group sheet_vectors by frequency
        freq_groups = defaultdict(list)
        for v_start, v_end, v_freq in sheet_vectors:
            freq_groups[v_freq].append((v_start, v_end))

        # Combine intervals with same frequency
        new_sheet_vectors = []
        combination_to_singles = {}
        for v_freq, intervals in freq_groups.items():
            intervals.sort()
            combined_intervals = []
            current_combination = [intervals[0]]

            for i in range(1, len(intervals)):
                prev_start, prev_end = current_combination[-1]
                curr_start, curr_end = intervals[i]

                if prev_end == curr_start:  # Valid combination
                    current_combination.append(intervals[i])
                else:
                    if len(current_combination) > 1:
                        combined_start = current_combination[0][0]
                        combined_end = current_combination[-1][1]
                        combined_intervals.append((combined_start, combined_end))
                        combination_to_singles[(combined_start, combined_end, v_freq)] = set(current_combination)
                    current_combination = [intervals[i]]

            if len(current_combination) > 1:
                combined_start = current_combination[0][0]
                combined_end = current_combination[-1][1]
                combined_intervals.append((combined_start, combined_end))
                combination_to_singles[(combined_start, combined_end, v_freq)] = set(current_combination)

            new_sheet_vectors.extend([(start, end, v_freq) for start, end in combined_intervals])
            new_sheet_vectors.extend([(start, end, v_freq) for start, end in intervals])

        results = []

        # Main processing loop
        for s_start, s_end, s_freq, s_num, (freq_min, freq_max) in summary:
            matching_vectors = [
                (v_start, v_end, v_freq) for v_start, v_end, v_freq in new_sheet_vectors
                if freq_min <= v_freq <= freq_max
            ]

            summary_length = s_end - s_start
            best_matches = []

            for v_start, v_end, v_freq in matching_vectors:
                sheet_vector_length = v_end - v_start
                ratio = summary_length / sheet_vector_length

                if 0.6 <= ratio <= 1.4:
                    midpoint_diff = abs(((v_start + v_end) / 2) - ((s_start + s_end) / 2))
                    overlap_start = max(s_start, v_start)
                    overlap_end = min(s_end, v_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    overlap_proportion = overlap_length / summary_length

                    if overlap_proportion > 0:
                        is_combination = (v_start, v_end, v_freq) in combination_to_singles
                        best_matches.append((v_start, v_end, v_freq, ratio, midpoint_diff, is_combination))

            if best_matches:
                best_match = min(best_matches, key=lambda x: x[4])  # Closest midpoint_diff
                results.append((
                    s_start, s_end, best_match[0], best_match[1], best_match[3], best_match[5], s_num, s_freq
                ))

        # Print all best matches before filtering
        print("\nAll Best Matches Before Filtering:")
        for result in results:
            print(f"  Summary Start={result[0]}, End={result[1]}, Match Start={result[2]}, End={result[3]}, "
                  f"Ratio={result[4]:.2f}, Is Combination={result[5]}, s_num={result[6]}, Summary Frequency={result[7]}")

        # Filter results
        filtered_results = [
            r for r in results if (not r[5] or 0.9 <= r[4] <= 1.1)
        ]

        if not filtered_results:
            print("\nNo filtered results found.")
            return None

        # Sort by s_num in descending order
        filtered_results.sort(key=lambda x: x[6], reverse=True)

        # Print filtered matches
        print("\nFiltered Best Matches:")
        for result in filtered_results:
            print(f"  Summary Start={result[0]}, End={result[1]}, Match Start={result[2]}, End={result[3]}, "
                  f"Ratio={result[4]:.2f}, Is Combination={result[5]}, s_num={result[6]}, Summary Frequency={result[7]}")

        # Use the highest s_num result for the first 4 rounds
        if len(transformed_intervals) < 4:
            print("\nReturning result based on largest s_num for initial rounds:")
            best_result = filtered_results[0]
            print(
                f"\033[94mSummary Start={best_result[0]}, End={best_result[1]}, Match Start={best_result[2]}, End={best_result[3]}, "
                f"Ratio={best_result[4]:.2f}, Is Combination={best_result[5]}, s_num={best_result[6]}, Summary Frequency={best_result[7]}\033[0m")
            return best_result[0:4]

        # Update filtered results with distances to transformed_intervals
        updated_filtered_results = []
        for result in filtered_results:
            s_middle = (result[1] + result[0]) / 2
            distances = [
                            abs(s_middle - t_start) for t_start, t_end in transformed_intervals
                        ] + [
                            abs(s_middle - t_end) for t_start, t_end in transformed_intervals
                        ]
            updated_filtered_results.append(result + (min(distances),))

        filtered_results = updated_filtered_results
        filtered_results.sort(key=lambda x: x[7], reverse=True)  # Sort by distance

        # Print filtered matches with distances
        print("\nFiltered Best Matches with Distances:")
        for result in filtered_results:
            print(f"  Summary Start={result[0]}, End={result[1]}, Match Start={result[2]}, End={result[3]}, "
                  f"Ratio={result[4]:.2f}, Is Combination={result[5]}, s_num={result[6]}, Summary Frequency={result[7]}, "
                  f"Distance to Nearest Transformed Interval={result[8]:.2f}")

        # Return the result with the greatest distance
        best_result = filtered_results[0]
        print("\nFinal Best Match Based on Distance:")
        print(
            f"\033[94mSummary Start={best_result[0]}, End={best_result[1]}, Match Start={best_result[2]}, End={best_result[3]}, "
            f"Ratio={best_result[4]:.2f}, Is Combination={best_result[5]}, s_num={best_result[6]}, Summary Frequency={best_result[7]}, "
            f"Distance to Nearest Transformed Interval={best_result[8]:.2f}\033[0m")
        return best_result[0:4]

    def process_transformation(summary, transformed_vectors, transformed_intervals):
        """
        Runs a single transformation iteration.
        """
        result = find_best_transform(summary, sheet_vectors, transformed_intervals)
        if not result:
            raise ValueError("No valid transformation found.")

        summary_start, summary_end, sheet_start, sheet_end = result
        print(
            f"Transformation: Summary Start={summary_start}, Summary End={summary_end}, Sheet Start={sheet_start}, Sheet End={sheet_end}")

        # Adjust vectors based on the transformation
        updated_transformed_vectors, updated_summary = adjust_vectors(
            sheet_start, sheet_end, summary_start, summary_end, summary, transformed_vectors, transformed_intervals
        )

        # Update summary by excluding intervals used in this transformation
        updated_summary = [
            s for s in updated_summary
            if not (s[0] < summary_end and s[1] > summary_start)
        ]

        # Add the interval to transformed intervals
        transformed_intervals.append((sheet_start, sheet_end))

        return updated_summary, updated_transformed_vectors, transformed_intervals

    transformed_intervals = []
    current_summary = summary

    for i in range(rounds):
        print(f"Running transformation round {i + 1}...")
        try:
            current_summary, transformed_audio_vectors, transformed_intervals = process_transformation(
                current_summary, transformed_audio_vectors, transformed_intervals
            )
        except ValueError as e:
            print(f"No more valid transformations available at round {i + 1}: {e}")
            break

        # Plot the results for this round
        plot_results(transformed_audio_vectors, sheet_vectors, f"Round {i + 1}")

    return transformed_audio_vectors


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
                num_points = len(vectors_in_box)
                freq_bounds = (freq_start, freq_end)
                summaries.append((min_time, max_time, avg_frequency, num_points, freq_bounds))

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

