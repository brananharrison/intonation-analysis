import numpy as np
import pandas as pd

from dynamic_time_warp import plot_results
from dynamic_time_warp import load_data, get_equal_temperament_frequencies, shift_and_scale_audio_vectors
from dynamic_time_warp import prepare_vectors, trim_vectors


import numpy as np

import numpy as np

import numpy as np

def run_dtw(transformed_audio, trimmed_sheet, time_radius=1, frequency_tolerance=0.02, extended_time_radius=2):
    path = []
    audio_mapped = set()
    sheet_mapped = set()

    for i, (a_time, a_value) in enumerate(transformed_audio):
        best_distance = float('inf')
        best_index = None

        for j, (s_time, s_value) in enumerate(trimmed_sheet):
            if abs(a_value - s_value) / max(a_value, s_value) <= frequency_tolerance:
                applicable_time_radius = extended_time_radius
            else:
                applicable_time_radius = time_radius

            if abs(a_time - s_time) <= applicable_time_radius:
                distance = np.sqrt((a_time - s_time) ** 2 + (a_value - s_value) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = j

        if best_index is not None:
            path.append((i, best_index))
            audio_mapped.add(i)
            sheet_mapped.add(best_index)

    for j, (s_time, s_value) in enumerate(trimmed_sheet):
        if j not in sheet_mapped:
            best_distance = float('inf')
            best_index = None

            for i, (a_time, a_value) in enumerate(transformed_audio):
                if abs(a_value - s_value) / max(a_value, s_value) <= frequency_tolerance:
                    applicable_time_radius = extended_time_radius
                else:
                    applicable_time_radius = time_radius

                if abs(a_time - s_time) <= applicable_time_radius:
                    distance = np.sqrt((a_time - s_time) ** 2 + (a_value - s_value) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        best_index = i

            if best_index is not None:
                path.append((best_index, j))
                audio_mapped.add(best_index)
                sheet_mapped.add(j)

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

def count_matches_within_radius(transformed_audio, trimmed_sheet, path, time_threshold=0.5, freq_threshold_ratio=0.1):
    match_count = 0
    for i, j in path:
        point_audio = transformed_audio[i]
        point_sheet = trimmed_sheet[j]

        # Calculate absolute differences
        time_diff = abs(point_audio[0] - point_sheet[0])
        freq_diff = abs(point_audio[1] - point_sheet[1])

        # Calculate frequency threshold as 10% of the sheet frequency
        freq_threshold = freq_threshold_ratio * abs(point_sheet[1])

        # Check if both conditions are satisfied
        if time_diff <= time_threshold and freq_diff <= freq_threshold:
            match_count += 1

    return match_count

frequencies = get_equal_temperament_frequencies()
audio_data_df, sheet_music_df = load_data("exports/audio_csv.csv", "exports/sheet_music_csv.csv")
audio_vectors, sheet_vectors = prepare_vectors(audio_data_df, sheet_music_df, frequencies)



audio_vectors = trim_vectors(audio_vectors, 23, 0)
audio_vectors = shift_and_scale_audio_vectors(audio_vectors, shift=-26, scale=1.05)

path = run_dtw(audio_vectors, sheet_vectors)
distance = calculate_total_distance_with_octave_correction(audio_vectors, sheet_vectors, path)
valid_points = map_valid_points_with_octave_correction(audio_vectors, sheet_vectors)
plot_results(audio_vectors, sheet_vectors, "Title", valid_points)

for idx, ((audio_time, audio_freq), (sheet_time, sheet_freq)) in enumerate(valid_points):
    time_distance = abs(audio_time - sheet_time)
    freq_distance = abs(audio_freq - sheet_freq)
    print(f"Index {idx}: Time Distance = {round(time_distance, 2)}, Frequency Distance = {round(freq_distance, 2)}")

match_count = count_matches_within_radius(audio_vectors, sheet_vectors, path)

print("Match count:", match_count)
print("Total distance:", round(distance, 2))
