import os

from intonation_app.dtw_functions import get_equal_temperament_frequencies, load_data, prepare_vectors, \
    find_optimal_transformation, shift_and_scale_audio_vectors, plot_results, map_valid_points, save_results, \
    map_points_onto_sheet_music, analyze_intonation, optimize_subsets, find_gaps, summarize_gaps, run_dtw


def map_frequency_vectors(audio_csv_path, sheet_csv_path, exports_dir="exports"):
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)

    # region Load and prepare data
    frequencies = get_equal_temperament_frequencies()
    audio_data_df, sheet_music_df = load_data(audio_csv_path, sheet_csv_path)
    raw_audio_vectors, sheet_vectors = prepare_vectors(audio_data_df, sheet_music_df, frequencies)

    # Plot un-normalized data
    plot_results(raw_audio_vectors, sheet_vectors, title="Un-normalized Audio and Sheet Music Alignment")

    # endregion

    # region Set optimization parameters
    audio_duration = raw_audio_vectors[-1][0] - raw_audio_vectors[0][0]
    sheet_duration = sheet_vectors[-1][1] - sheet_vectors[0][0]

    min_scale_range = 0.5
    max_scale_range = 1.5
    shift_range = (-audio_duration, sheet_duration)

    # endregion

    # region Rough optimization
    print("Scale range:", (min_scale_range, max_scale_range), "Step:", 0.1)
    print("Shift range:", shift_range, "Step:", 5)
    transformation_results = find_optimal_transformation(
        raw_audio_vectors, sheet_vectors,
        scale_range=(0.5, 1.5), scale_step=0.1,
        shift_range=shift_range, shift_step=1)

    optimal_shift = transformation_results["optimal_shift"]
    optimal_scale = transformation_results["optimal_scale"]
    best_mapping = transformation_results["best_mapping"]
    unmapped_points = transformation_results["unmapped_points"]

    print(f"Rough optimization mapped points: \033[92m{(1 - len(unmapped_points) / len(raw_audio_vectors)) * 100:.2f}%\033[0m")
    print(f"Sheet music points: \033[92m{len(best_mapping)}\033[0m")

    # endregion

    # region Precise optimization
    print("Scale range:", (min_scale_range, max_scale_range), "Step:", 0.1)
    print("Shift range:", shift_range, "Step:", 1)
    transformation_results = find_optimal_transformation(
        raw_audio_vectors, sheet_vectors,
        scale_range=(optimal_scale - 0.1, optimal_scale + 0.1), scale_step=0.1,
        shift_range=(optimal_shift - 3, optimal_shift + 3), shift_step=0.25)

    optimal_shift = transformation_results["optimal_shift"]
    optimal_scale = transformation_results["optimal_scale"]
    best_mapping = transformation_results["best_mapping"]
    unmapped_points = transformation_results["unmapped_points"]

    print(f"Precise optimization mapped points: \033[92m{(1 - len(unmapped_points) / len(raw_audio_vectors)) * 100:.2f}%\033[0m")
    print(f"Sheet music points: \033[92m{len(best_mapping)}\033[0m")

    # endregion

    # region Transform vectors and plot
    transformed_audio_vectors = shift_and_scale_audio_vectors(raw_audio_vectors, shift=optimal_shift, scale=optimal_scale)
    gaps = find_gaps(transformed_audio_vectors)

    #plot_results(unmapped_points, sheet_vectors, "Unmapped transformed vs sheet vectors")
    plot_results(transformed_audio_vectors, sheet_vectors, "Plot without gaps", valid_points=None)
    plot_results(transformed_audio_vectors, sheet_vectors, "Plot with gaps", valid_points=None, gaps=gaps)


    summary = summarize_gaps(gaps, transformed_audio_vectors)

    refined_audio_vectors = optimize_subsets(transformed_audio_vectors, sheet_vectors, summary)
    plot_results(refined_audio_vectors, sheet_vectors, "Subset audio vectors vs sheet vectors")

    mapping, unmapped_points = run_dtw(refined_audio_vectors, sheet_vectors)
    print(
        f"Subset optimization mapped points: \033[92m{(1 - len(unmapped_points) / len(raw_audio_vectors)) * 100:.2f}%\033[0m")
    print(f"Sheet music points: \033[92m{len(best_mapping)}\033[0m")
    plot_results(unmapped_points, sheet_vectors, "Unmapped subsets vs sheet vectors")

    # endregion


    # Map vectors with octave correction
    valid_points = map_valid_points(mapping)

    # Save original points to CSV
    save_results(os.path.join(exports_dir, 'valid_points.csv'), valid_points)

    # Construct vectors with applied octave correction
    adjusted_audio_vectors = [(audio[0], audio[1]) for audio, _ in valid_points]
    adjusted_sheet_vectors = [(sheet[0], sheet[1], sheet[2]) for _, sheet in valid_points]

    # Plot results
    plot_results(adjusted_audio_vectors, adjusted_sheet_vectors, "Mapped Valid Points", valid_points)

    # Map points onto sheet music
    valid_points_df = map_points_onto_sheet_music(valid_points)
    valid_points_df.to_csv(os.path.join(exports_dir, 'points_mapped_to_sheet_music.csv'), index=False)

    # Analyze intonation errors
    intonation = analyze_intonation(valid_points_df)
    intonation.to_csv(os.path.join(exports_dir, 'processed_intonation.csv'), index=False)

    print(f"Aggregated results saved to: {os.path.join(exports_dir, 'one_to_one_mapping_to_sheet_music.csv')}")

    print(f"Results saved to {os.path.join(exports_dir, 'original_intonation_errors.csv')}")
