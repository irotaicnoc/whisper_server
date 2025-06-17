import os
import logging
import tempfile
from flask import Flask, request, jsonify

import whisper


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the Whisper model once when the server starts
# You can change 'base' to 'small', 'medium', 'large', etc., depending on your needs and system resources.
try:
    logging.info('Loading OpenAI Whisper model. This may take a moment...')
    model = whisper.load_model('turbo')
    # model = whisper.load_model('large')
    logging.info('Whisper model loaded successfully.')
except Exception as e:
    logging.error(f'Error loading Whisper model: {e}')
    model = None


@app.route(rule='/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Receives an audio file, transcribes it using OpenAI Whisper,
    and returns the transcription.
    """
    if model is None:
        return jsonify({'error': 'Whisper model failed to load. Please check server logs.'}), 500

    # Check if an audio file is present in the request
    if 'audio' not in request.files:
        logging.warning('No "audio" file part in the request.')
        return jsonify({'error': 'No audio file part in the request'}), 400

    audio_file = request.files['audio']

    # Check if the filename is empty
    if audio_file.filename == '':
        logging.warning('No selected audio file.')
        return jsonify({'error': 'No selected audio file'}), 400

    if audio_file:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        logging.info(f'Received audio file: {audio_file.filename}. Saved temporarily to {temp_audio_path}')

        try:
            # Transcribe the audio using Whisper
            logging.info('Starting transcription...')
            result = model.transcribe(audio=temp_audio_path)
            transcription = result['text']
            logging.info('Transcription completed.')
            return jsonify({'transcription': transcription}), 200
        except Exception as e:
            logging.error(f'Error during transcription: {e}')
            return jsonify({'error': f'Error during transcription: {str(e)}'}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logging.info(f'Temporary file {temp_audio_path} deleted.')
    else:
        logging.warning('Unexpected error: Audio file not processed.')
        return jsonify({'error': 'Unexpected error processing audio file'}), 500


if __name__ == '__main__':
    logging.info('Starting Flask server...')
    # You can change the host to '0.0.0.0' to make it accessible from your robot on the same network.
    # Be cautious when exposing to the internet.
    app.run(host='0.0.0.0', port=5000)
    logging.info('Flask server stopped.')
