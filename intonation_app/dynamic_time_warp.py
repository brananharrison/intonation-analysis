import os

from intonation_app.dtw_functions import get_equal_temperament_frequencies, load_data, prepare_vectors, \
    find_optimal_transformation, shift_and_scale_audio_vectors, plot_results, map_valid_points, save_results, \
    map_points_onto_sheet_music, parse_single_null_values_using_audio_json, interpolate_doubled_notes_with_audio_json, \
    analyze_intonation


def map_frequency_vectors(audio_csv_path, sheet_csv_path, exports_dir="exports"):
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)

    # Load and prepare data
    frequencies = get_equal_temperament_frequencies()
    audio_data_df, sheet_music_df = load_data(audio_csv_path, sheet_csv_path)
    raw_audio_vectors, sheet_vectors = prepare_vectors(audio_data_df, sheet_music_df, frequencies)

    # Plot un-normalized data
    #plot_results(raw_audio_vectors, sheet_vectors, title="Un-normalized Audio and Sheet Music Alignment")

    # Optimize transformation
    audio_duration = raw_audio_vectors[-1][0] - raw_audio_vectors[0][0]
    sheet_duration = sheet_vectors[-1][0] - sheet_vectors[0][0]

    min_scale_range = min(sheet_duration / audio_duration, 0.5)
    max_scale_range = max(sheet_duration / (audio_duration / 2), 1.5)
    shift_range = (-audio_duration, sheet_duration)

    # Rough optimization
    print("Scale range:", (min_scale_range, max_scale_range), "Step:", 0.1)
    print("Shift range:", shift_range, "Step:", 5)
    transformation_results = find_optimal_transformation(
        raw_audio_vectors, sheet_vectors,
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
        raw_audio_vectors, sheet_vectors,
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
    transformed_audio_vectors = shift_and_scale_audio_vectors(raw_audio_vectors, shift=optimal_shift, scale=optimal_scale)
    plot_results(transformed_audio_vectors, sheet_vectors, "Transformed audio vectors vs sheet vectors")

    # Map vectors with octave correction
    valid_points = map_valid_points(transformed_audio_vectors, sheet_vectors)

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

    # Parse remaining single-null values with audio_json, or remove the row if no match
    valid_points_without_null_df = parse_single_null_values_using_audio_json(valid_points_df, "exports/audio_json.json", optimal_shift, optimal_scale)

    # Handles the case where algo didn't detect the break in a doubled note
    processed_points = interpolate_doubled_notes_with_audio_json(valid_points_without_null_df, optimal_shift, optimal_scale, "exports/audio_json.json", "exports/audio_csv.csv")

    # Analyze intonation errors
    intonation = analyze_intonation(processed_points)

    intonation.to_csv(os.path.join(exports_dir, 'processed_intonation.csv'), index=False)

    print(f"Aggregated results saved to: {os.path.join(exports_dir, 'one_to_one_mapping_to_sheet_music.csv')}")

    print(f"Results saved to {os.path.join(exports_dir, 'original_intonation_errors.csv')}")
