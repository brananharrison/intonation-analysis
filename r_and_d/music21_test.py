import pandas as pd
from music21 import converter, note, chord, stream
import os


mxl_file_path = "/Users/branan/Downloads/bach_andante.mxl"
print(os.path.exists(mxl_file_path))

score = converter.parse(mxl_file_path)
print(score)

# Initialize variables
cumulative_time = 0  # Tracks cumulative time across measures
data = []  # List to store DataFrame rows

# Iterate through the parts in the score
for part in score.parts:
    for measure in part.getElementsByClass('Measure'):
        measure_time_indices = {}  # Dictionary to track time_index for all voices in the measure

        # Check if there are voices in the measure
        voices = measure.getElementsByClass(stream.Voice)
        if voices:
            for voice_index, voice in enumerate(voices, start=1):
                current_time = 0  # Local time within the measure

                for element in voice:
                    if isinstance(element, (note.Note, chord.Chord)):
                        duration = element.duration.quarterLength
                        start_time = cumulative_time + current_time
                        end_time = start_time + duration

                        if isinstance(element, note.Note):
                            notes = [element.nameWithOctave]
                        elif isinstance(element, chord.Chord):
                            notes = [n.nameWithOctave for n in element.notes]

                        for single_note in notes:
                            data.append({
                                "start_time": start_time,
                                "end_time": end_time,
                                "note": single_note
                            })

                        current_time += duration
                    elif isinstance(element, note.Rest):
                        duration = element.duration.quarterLength
                        current_time += duration

        else:
            current_time = 0  # Local time within the measure
            for element in measure.notesAndRests:
                if isinstance(element, (note.Note, chord.Chord)):
                    duration = element.duration.quarterLength
                    start_time = cumulative_time + current_time
                    end_time = start_time + duration

                    if isinstance(element, note.Note):
                        notes = [element.nameWithOctave]
                    elif isinstance(element, chord.Chord):
                        notes = [n.nameWithOctave for n in element.notes]

                    for single_note in notes:
                        data.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "note": single_note
                        })

                    current_time += duration
                elif isinstance(element, note.Rest):
                    duration = element.duration.quarterLength
                    current_time += duration

        cumulative_time += measure.quarterLength

# Create DataFrame
columns = ["start_time", "end_time", "note"]
df = pd.DataFrame(data, columns=columns)

# Sort DataFrame by start_time ascending and secondarily by end_time ascending
df = df.sort_values(by=["start_time", "end_time"])

# Define the directory and the output file path
output_dir = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "sheet_music_csv.csv")

# Save DataFrame to CSV
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
