import json
import pandas as pd

def json_to_time_indexed_table(json_file_path, output_csv_path, appearance_threshold):
    """
    Processes note frequencies from a JSON file and saves the results to a CSV file.

    Args:
        json_file_path (str): Path to the input JSON file.
        output_csv_path (str): Path to save the output CSV file.
        appearance_threshold (int): Number of consecutive appearances required to count a note.
            Set to 1 to count notes immediately.
    """
    # Load JSON data
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Data structures
    current_notes = {}  # Tracks ongoing notes
    prepare_data_frame = []  # Notes ready to be added to the DataFrame
    pending_notes = {}  # Tracks notes that need consecutive appearances to count

    # Process each entry
    for entry in json_data:
        start_time = entry["start_time"]
        end_time = entry["end_time"]
        frequencies = entry["frequency"]
        notes = entry["notes"]

        # Track notes seen in the current entry
        current_entry_notes = {}

        for idx, note in enumerate(notes):
            note_name = note["Name"]
            frequency = frequencies[idx]  # Match frequency by index

            if appearance_threshold == 1:
                # Immediate counting logic
                if note_name in current_notes:
                    # Update the ongoing note
                    current_note = current_notes[note_name]
                    current_note["frequency_sum"] += frequency
                    current_note["count"] += 1
                    current_note["end_time"] = end_time
                else:
                    # Add new note to current_notes
                    current_notes[note_name] = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "frequency_sum": frequency,
                        "count": 1
                    }
            else:
                # Threshold-based counting logic
                if note_name in current_notes:
                    # Update the ongoing note
                    current_note = current_notes[note_name]
                    current_note["frequency_sum"] += frequency
                    current_note["count"] += 1
                    current_note["end_time"] = end_time
                elif note_name in pending_notes:
                    # Increment the consecutive count
                    pending_notes[note_name]["consecutive_count"] += 1

                    # Promote to current_notes if threshold is met
                    if pending_notes[note_name]["consecutive_count"] >= appearance_threshold:
                        current_notes[note_name] = pending_notes.pop(note_name)
                        current_notes[note_name]["frequency_sum"] += frequency
                        current_notes[note_name]["count"] += 1
                        current_notes[note_name]["end_time"] = end_time
                else:
                    # Add new note to pending_notes
                    pending_notes[note_name] = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "frequency_sum": frequency,
                        "count": 1,
                        "consecutive_count": 1
                    }

            current_entry_notes[note_name] = True

        # Handle notes not present in the current entry
        notes_to_remove = []
        for note_name, note_data in current_notes.items():
            if note_name not in current_entry_notes:
                # Finalize the note
                average_frequency = note_data["frequency_sum"] / note_data["count"]
                prepare_data_frame.append({
                    "start_time": note_data["start_time"],
                    "end_time": note_data["end_time"],
                    "note": note_name,
                    "average_frequency": average_frequency
                })
                notes_to_remove.append(note_name)

        # Remove finalized notes from current_notes
        for note_name in notes_to_remove:
            del current_notes[note_name]

        # Handle notes in pending_notes that did not appear in this entry
        if appearance_threshold > 0:
            pending_notes_to_remove = []
            for note_name, note_data in pending_notes.items():
                if note_name not in current_entry_notes:
                    pending_notes_to_remove.append(note_name)

            for note_name in pending_notes_to_remove:
                del pending_notes[note_name]

    # Finalize remaining notes in current_notes
    for note_name, note_data in current_notes.items():
        average_frequency = note_data["frequency_sum"] / note_data["count"]
        prepare_data_frame.append({
            "start_time": note_data["start_time"],
            "end_time": note_data["end_time"],
            "note": note_name,
            "average_frequency": average_frequency
        })

    # Prepare the DataFrame and sort by start_time
    final_notes = pd.DataFrame(prepare_data_frame)
    final_notes = final_notes.sort_values(by="start_time")
    final_notes.to_csv(output_csv_path, index=False)

    print(f"Processed data saved to {output_csv_path}")

