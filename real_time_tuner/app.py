import eventlet
eventlet.monkey_patch()  # Must be called before importing other modules

import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from r_and_d.functions import (
    find_all_spikes,
    group_frequencies,
    find_all_fundamental_frequencies,
    analyze_intervals, live_zero_padded_fourier_transform, update_fundamental,
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)

app = Flask(__name__)
socketio = SocketIO(app)  # Enable WebSocket support with Flask-SocketIO

# Global variables for persisting state
previous_intonation = {}
previous_fundamentals = []
previous_note_name = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    global previous_fundamentals, previous_intonation, previous_note_name

    try:
        # Receive audio data, frame rate, and metadata
        payload = request.json
        audio_data = payload.get('audio', [])
        frame_rate = payload.get('frame_rate', 44100)
        start_time = payload.get('start_time')
        end_time = payload.get('end_time')
        time_of_sending = payload.get('time_of_sending')

        if not audio_data:
            logging.warning("No audio data received.")
            return jsonify({"error": "No audio data received."}), 400

        # Log metadata and audio data length for debugging
        logging.info(f"Audio received: Start={start_time}, End={end_time}, Sent={time_of_sending}")
        logging.info(f"Audio data length: {len(audio_data)} samples")

        # Convert audio to numpy array
        audio_array = np.array(audio_data)

        # Start processing
        start_time_processing = datetime.now()
        frequencies, amplitude = live_zero_padded_fourier_transform(
            audio_array, frame_rate, scale_n_by=10
        )

        if amplitude.any() and frequencies.any():
            spikes = find_all_spikes(frequencies, amplitude)
            grouped_freqs = group_frequencies(spikes)
            fundamentals = find_all_fundamental_frequencies(grouped_freqs)
            intonation = analyze_intervals(fundamentals)

            # Update previous data only if valid fundamentals are found
            if intonation:
                previous_fundamentals = fundamentals
                previous_intonation = intonation
                previous_note_name = intonation.get("Name")
        else:
            # No valid data; reuse previous values
            fundamentals = previous_fundamentals
            intonation = previous_intonation

        # Log processing duration
        processing_duration = (datetime.now() - start_time_processing).total_seconds()
        logging.info(f"Processing duration: {processing_duration:.3f} seconds")
        logging.debug(f"Previous Fundamentals: {previous_fundamentals}")
        logging.debug(f"Previous Intonation: {previous_intonation}")

        # Prepare the response
        response_data = {
            "metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "time_of_sending": time_of_sending,
                "processing_duration": processing_duration,
                "response_time": datetime.now().isoformat(),
            },
            "results": {
                "frequencies": fundamentals if fundamentals else previous_fundamentals,
                "intonation": intonation if intonation else previous_intonation,
            }
        }
        return jsonify(response_data)

    except Exception as e:
        logging.exception("Error processing audio data:")
        return jsonify({"error": str(e)}), 500

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """
    WebSocket endpoint for processing audio streams in real-time.
    """
    try:
        # Extract payload directly from WebSocket data
        audio_data = data.get('audio', [])
        frame_rate = data.get('frame_rate', 44100)
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        time_of_sending = datetime.now().isoformat()

        # Log audio data length for debugging
        logging.info(f"WebSocket audio data length: {len(audio_data)} samples")

        payload = {
            "audio": audio_data,
            "frame_rate": frame_rate,
            "start_time": start_time,
            "end_time": end_time,
            "time_of_sending": time_of_sending,
        }

        # Call analyze_audio logic directly with payload
        response = analyze_audio_payload(payload)
        emit('audio_response', response)

    except Exception as e:
        logging.exception("Error in WebSocket audio stream:")
        emit('error', {"error": str(e)})

def analyze_audio_payload(payload):
    """
    Refactored audio analysis logic to process payloads outside the Flask request context.
    """
    global previous_fundamentals, previous_intonation, previous_note_name

    audio_data = payload.get('audio', [])
    frame_rate = payload.get('frame_rate', 44100)

    # Log audio data length for debugging
    logging.info(f"Payload audio data length: {len(audio_data)} samples")

    # Ensure this mirrors analyze_audio() but uses the given payload
    audio_array = np.array(audio_data)
    frequencies, amplitude = live_zero_padded_fourier_transform(audio_array, frame_rate, scale_n_by=10)

    if amplitude.any() and frequencies.any():
        spikes = find_all_spikes(frequencies, amplitude)
        grouped_freqs = group_frequencies(spikes)
        fundamentals = find_all_fundamental_frequencies(grouped_freqs)
        intonation = analyze_intervals(fundamentals)

        if intonation:
            # Update previous data only if valid fundamentals are found
            previous_fundamentals = fundamentals
            previous_intonation = intonation
            previous_note_name = intonation.get("Name")
    else:
        # No valid data; reuse previous values
        fundamentals = previous_fundamentals
        intonation = previous_intonation

    return {
        "metadata": {
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "time_of_sending": payload.get("time_of_sending"),
        },
        "results": {
            "frequencies": fundamentals,
            "intonation": intonation,
        },
    }

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
