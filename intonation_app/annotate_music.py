import csv
from music21 import converter, environment, stream, note
from PIL import Image, ImageOps
import glob
import os
import shutil

def annotate_sheet_music(musicxml_file, csv_file_path):
    # Set MuseScore path
    environment.set('musescoreDirectPNGPath', '/Applications/MuseScore 4.app/Contents/MacOS/mscore')

    # Create a directory for temporary MuseScore files
    musescore_dir = 'musescore_exports'
    os.makedirs(musescore_dir, exist_ok=True)

    # Create the exports directory if it doesn't exist
    exports_dir = 'exports'
    os.makedirs(exports_dir, exist_ok=True)

    score = converter.parse(musicxml_file)

    intonation_map = {}

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            offset, expected_freq, actual_freq, deviation = map(float, row)
            intonation_map[round(offset, 2)] = round(deviation, 2)

    def calculate_global_offset(element):
        total_offset = element.offset
        parent = element.activeSite
        while parent is not None and not isinstance(parent, stream.Score):
            total_offset += parent.offset
            parent = parent.activeSite
        return round(total_offset, 2)

    for part in score.parts:
        for n in part.recurse().notes:
            offset = calculate_global_offset(n)  # Get global offset recursively
            if offset in intonation_map:
                deviation = intonation_map[offset]
                annotation = f"{deviation:.1f}"

                # Clear existing lyrics to avoid duplicates
                n.lyrics = []

                # Add lyric with applyRaw=True to properly handle the '-' sign
                n.addLyric(annotation, applyRaw=True)

                # Adjust color styling based on the deviation
                lyric = n.lyrics[-1]  # Access the most recently added lyric
                lyric.style.color = '#FF6666' if abs(deviation) > 5 else 'green'

    # Save annotated music XML file in the musescore_exports directory
    annotated_file = os.path.join(musescore_dir, 'Twinkle_twinle_annotated.xml')
    score.write('musicxml', fp=annotated_file)

    # Generate PNG file and save it in the musescore_exports directory
    png_file = os.path.join(musescore_dir, 'Twinkle_twinle_annotated.png')
    score.write('musicxml.png', fp=png_file)

    # Search for the generated PNG files
    png_file_pattern = os.path.join(musescore_dir, 'Twinkle_twinle_annotated*.png')
    png_files = glob.glob(png_file_pattern)
    if not png_files:
        raise FileNotFoundError(f"No file matching {png_file_pattern} was generated.")

    # Process the image with margins
    output_png = os.path.join(exports_dir, 'annotated_sheet_music.png')

    image = Image.open(max(png_files, key=os.path.getctime))
    image_with_margins = ImageOps.expand(image, border=150, fill='white')
    image_with_margins.save(output_png)

    print(f"PNG with margins saved to {output_png}")

    # Cleanup: Delete the musescore_exports directory and its contents
    shutil.rmtree(musescore_dir)
    print(f"Temporary directory {musescore_dir} deleted.")
