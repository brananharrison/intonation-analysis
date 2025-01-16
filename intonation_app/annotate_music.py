import csv
from music21 import converter, environment, stream
from PIL import Image, ImageOps
import glob
import os
import shutil

def annotate_sheet_music(musicxml_file, csv_file_path):
    # Set MuseScore path
    environment.set('musescoreDirectPNGPath', '/Applications/MuseScore 4.app/Contents/MacOS/mscore')

    # Create directories for temporary MuseScore files and exports
    musescore_dir = 'musescore_exports'
    os.makedirs(musescore_dir, exist_ok=True)
    exports_dir = 'exports'
    os.makedirs(exports_dir, exist_ok=True)

    score = converter.parse(musicxml_file)

    # Initialize an intonation map to store deviations for each offset
    intonation_map = {}
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                offset = float(row[0]) if row[0] else None
                expected_freq = float(row[1]) if row[1] else None
                deviation = float(row[3]) if row[3] else None

                if offset is not None and deviation is not None and expected_freq is not None:
                    if offset not in intonation_map:
                        intonation_map[offset] = []
                    intonation_map[offset].append((expected_freq, deviation))
            except ValueError:
                print(f"Skipping row due to invalid data: {row}")

    def calculate_global_offset(element):
        total_offset = element.offset
        parent = element.activeSite
        while parent is not None and not isinstance(parent, stream.Score):
            total_offset += parent.offset
            parent = parent.activeSite
        return round(total_offset, 2)

    for part in score.parts:
        for n in part.recurse().notes:
            offset = calculate_global_offset(n)
            if offset in intonation_map:
                # Sort by frequency for stacking
                matches = sorted(
                    intonation_map[offset],
                    key=lambda freq_dev: freq_dev[0],
                    reverse=True  # Higher frequencies appear first
                )

                # Annotate each note
                n.lyrics = []  # Clear existing lyrics
                for idx, (expected_freq, deviation) in enumerate(matches):
                    if abs(n.pitch.frequency - expected_freq) > 5:  # Frequency tolerance
                        continue
                    annotation = f"{deviation:.1f}"
                    n.insertLyric(annotation, applyRaw=True)
                    lyric = n.lyrics[-1]

                    # Adjust vertical stacking
                    if lyric.style.relativeY is None:
                        lyric.style.relativeY = 0
                    lyric.style.relativeY += idx * 10

                    lyric.style.fontSize = 5

                    # Color styling based on deviation
                    lyric.style.color = '#FF6666' if abs(deviation) > 5 else 'green'

    # Save annotated MusicXML file
    annotated_file = os.path.join(musescore_dir, 'annotated.xml')
    score.write('musicxml', fp=annotated_file)

    # Generate PNGs using MuseScore
    score.write('musicxml.png', fp=os.path.join(musescore_dir, 'annotated'))
    png_file_pattern = os.path.join(musescore_dir, 'annotated*.png')
    png_files = sorted(glob.glob(png_file_pattern))
    if not png_files:
        raise FileNotFoundError(f"No PNG files matching {png_file_pattern} were generated.")

    # Combine PNGs into a single PDF
    images = [Image.open(png) for png in png_files]
    image_with_margins = [ImageOps.expand(image, border=150, fill='white') for image in images]
    output_pdf = os.path.join(exports_dir, 'annotated_sheet_music_combined.pdf')
    image_with_margins[0].save(output_pdf, save_all=True, append_images=image_with_margins[1:])

    print(f"Combined PDF saved to {output_pdf}")

    # Cleanup
    #shutil.rmtree(musescore_dir)
    #print(f"Temporary directory {musescore_dir} deleted.")
