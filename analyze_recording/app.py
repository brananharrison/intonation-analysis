from flask import Flask, render_template, send_from_directory, send_file
import os
import json

from r_and_d.functions import identify_notes, extract_audio_segment_for_localhost

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
FREQUENCIES_FILE = os.path.join(BASE_DIR, "audio_json.json")

@app.route('/')
def index():
    try:
        with open(FREQUENCIES_FILE, "r") as f:
            frequencies_data = json.load(f)
    except FileNotFoundError:
        return "Error: Frequencies file not found.", 404
    except json.JSONDecodeError:
        return "Error: Invalid JSON format in frequencies file.", 400

    plots = [f"{item['plot']}.png" for item in frequencies_data if "plot" in item]
    frequencies = [item["frequency"] for item in frequencies_data if "frequency" in item]
    notes = [identify_notes(item["frequency"]) for item in frequencies_data if "frequency" in item]
    return render_template('index.html', plots=plots, frequencies=frequencies, notes=notes)


@app.route('/exports/<path:filename>')
def exports(filename):
    return send_from_directory(EXPORTS_DIR, filename)



@app.route('/audio')
def audio():
    output_file_path = extract_audio_segment_for_localhost(
        "/Users/branan/Downloads/How to play Twinkle Twinkle on the violin!.wav",
        "/Users/branan/Downloads/Twinkle_excerpt.wav",
        50, 95
    )

    if not output_file_path or not os.path.exists(output_file_path):
        return "Error: Failed to generate audio file.", 500

    return send_file(output_file_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
