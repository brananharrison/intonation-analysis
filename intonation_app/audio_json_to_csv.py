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
    groups = {}  # Tracks note groups
    group_id_counter = 1  # Unique ID for each group

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
            frequency = frequencies[idx]

            if note_name in current_notes:
                # Update the ongoing note
                current_note = current_notes[note_name]
                current_note["frequency_sum"] += frequency
                current_note["count"] += 1
                current_note["end_time"] = end_time
                current_entry_notes[note_name] = True
            else:
                # Create a new note group if not present in current_notes
                current_notes[note_name] = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "frequency_sum": frequency,
                    "count": 1,
                    "group_id": group_id_counter,
                }
                groups[group_id_counter] = {
                    "note": note_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "frequency_sum": frequency,
                    "count": 1,
                }
                group_id_counter += 1
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
                    "average_frequency": average_frequency,
                    "note_count": note_data["count"],
                })
                # Update the group
                group_id = note_data["group_id"]
                groups[group_id]["end_time"] = note_data["end_time"]
                groups[group_id]["frequency_sum"] = note_data["frequency_sum"]
                groups[group_id]["count"] = note_data["count"]

                notes_to_remove.append(note_name)

        # Remove finalized notes from current_notes
        for note_name in notes_to_remove:
            del current_notes[note_name]

    # Finalize remaining notes in current_notes
    for note_name, note_data in current_notes.items():
        average_frequency = note_data["frequency_sum"] / note_data["count"]
        prepare_data_frame.append({
            "start_time": note_data["start_time"],
            "end_time": note_data["end_time"],
            "note": note_name,
            "average_frequency": average_frequency,
            "note_count": note_data["count"],
        })
        # Update the group
        group_id = note_data["group_id"]
        groups[group_id]["end_time"] = note_data["end_time"]
        groups[group_id]["frequency_sum"] = note_data["frequency_sum"]
        groups[group_id]["count"] = note_data["count"]

    # Prepare the DataFrame
    groups_df = pd.DataFrame(prepare_data_frame).sort_values(by="start_time")

    # Filter DataFrame based on note_count_limit
    groups_df = groups_df[groups_df["note_count"] > note_count_limit]

    # Look for potential merges after filtering
    merged_data = []
    i = 0
    while i < len(groups_df):
        current_row = groups_df.iloc[i]

        if i < len(groups_df) - 1:
            next_row = groups_df.iloc[i + 1]

            if (current_row["note"] == next_row["note"] and
                abs(current_row["end_time"] - next_row["start_time"]) <= 0.15):
                # Merge rows
                total_count = current_row["note_count"] + next_row["note_count"]
                new_avg_frequency = (
                    current_row["average_frequency"] * current_row["note_count"] +
                    next_row["average_frequency"] * next_row["note_count"]
                ) / total_count

                merged_row = {
                    "start_time": current_row["start_time"],
                    "end_time": next_row["end_time"],
                    "note": current_row["note"],
                    "average_frequency": new_avg_frequency,
                    "note_count": total_count,
                }
                # Add merged row and skip next row
                merged_data.append(merged_row)
                i += 2
                continue

        # If no merge, add the current row
        merged_data.append(current_row.to_dict())
        i += 1

    # Create DataFrame after merging
    merged_df = pd.DataFrame(merged_data)

    # Save the final DataFrame
    merged_df.to_csv(output_csv_path, index=False)

    print(f"Final merged notes saved to {output_csv_path}")

# Example usage
# json_to_time_indexed_table("input.json", "audio_csv.csv", note_count_limit=1)
