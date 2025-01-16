import json
import pandas as pd

def json_to_time_indexed_table(json_file_path, output_csv_path, note_count_limit=1):
    """
    Processes note frequencies from a JSON file and saves the results to a single CSV file.

    Args:
        json_file_path (str): Path to the input JSON file.
        output_csv_path (str): Path to save the final CSV file.
        note_count_limit (int): Minimum number of notes in a group to be included in the output.
    """
    # Load JSON data
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Data structures
    current_notes = {}  # Tracks ongoing notes
    prepare_data_frame = []  # Notes ready to be added to the DataFrame

    # Process each entry
    for entry in json_data:
        start_time = round(entry["start_time"], 3)
        end_time = round(entry["end_time"], 3)
        frequencies = entry["frequency"]
        notes = entry["notes"]

        # Track notes seen in the current entry
        current_entry_notes = {}

        for idx, note in enumerate(notes):
            note_name = note["Name"]
            frequency = frequencies[idx]

            if note_name in current_notes:
                # Update the ongoing note
                current_note = current_notes[note_name]
                current_note["frequencies"].append(frequency)
                current_note["end_times"].append(end_time)  # Track end time of each note occurrence
                current_note["start_times"].append(start_time)
                current_note["count"] += 1
                current_entry_notes[note_name] = True
            else:
                # Create a new note group if not present in current_notes
                current_notes[note_name] = {
                    "start_times": [start_time],
                    "end_times": [end_time],
                    "frequencies": [frequency],
                    "count": 1
                }
                current_entry_notes[note_name] = True

        # Handle notes not present in the current entry
        notes_to_remove = []
        for note_name, note_data in current_notes.items():
            if note_name not in current_entry_notes:
                # Finalize the note
                for i, start_time in enumerate(note_data["start_times"]):
                    prepare_data_frame.append({
                        "start_time": start_time,
                        "end_time": note_data["end_times"][i],
                        "note": note_name,
                        "frequency": note_data["frequencies"][i],
                        "note_count": note_data["count"]
                    })
                notes_to_remove.append(note_name)

        # Remove finalized notes from current_notes
        for note_name in notes_to_remove:
            del current_notes[note_name]

    # Finalize remaining notes in current_notes
    for note_name, note_data in current_notes.items():
        for i, start_time in enumerate(note_data["start_times"]):
            prepare_data_frame.append({
                "start_time": start_time,
                "end_time": note_data["end_times"][i],
                "note": note_name,
                "frequency": note_data["frequencies"][i],
                "note_count": note_data["count"]
            })

    # Prepare the DataFrame
    groups_df = pd.DataFrame(prepare_data_frame).sort_values(by="start_time")

    # Filter DataFrame based on note_count_limit
    groups_df = groups_df[groups_df["note_count"] > note_count_limit]

    # Save the final DataFrame
    groups_df.to_csv(output_csv_path, index=False)

    print(f"Final processed notes saved to {output_csv_path}")

# Example usage
# json_to_time_indexed_table("input.json", "audio_csv.csv", note_count_limit=1)
