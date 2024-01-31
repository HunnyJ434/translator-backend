# import requests
# from tensorflow.keras.models import load_model
# from io import BytesIO
# import tempfile
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Replace 'your_modified_link' with the modified link obtained from Google Drive
# google_drive_link = 'https://drive.google.com/uc?id=1wwAzakr6Se9tAeh1tC60IEaYdL30oBPc'

# print("Downloading the model...")

# response = requests.get(google_drive_link, stream=True)
# total_size = int(response.headers.get('content-length', 0))

# if response.status_code == 200:
#     model_content = BytesIO()
#     downloaded_size = 0
#     for data in response.iter_content(chunk_size=1024):
#         downloaded_size += len(data)
#         model_content.write(data)
#         done = int(50 * downloaded_size / total_size)
#         print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_size}/{total_size} bytes downloaded", end='', flush=True)

#     print("\nDownload complete.")
    
#     # Save the BytesIO content to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
#         temp_file.write(model_content.getvalue())
#         temp_file_path = temp_file.name

#     # Load the model from the temporary file
#     loaded_model = load_model(temp_file_path)
#     print("Model loaded successfully.")

#     # Clean up the temporary file
#     os.remove(temp_file_path)
# else:
#     print(f"\nFailed to download the model. Status code: {response.status_code}")

import requests
from tensorflow.keras.models import load_model, save_model
from io import BytesIO
import tempfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Replace 'your_modified_link' with the modified link obtained from Google Drive
google_drive_link = 'https://drive.google.com/uc?id=1wwAzakr6Se9tAeh1tC60IEaYdL30oBPc'
model_filename = 'your_model.h5'  # Choose a suitable filename

def download_model():
    print("Downloading the model...")

    response = requests.get(google_drive_link, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if response.status_code == 200:
        model_content = BytesIO()
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            downloaded_size += len(data)
            model_content.write(data)
            done = int(50 * downloaded_size / total_size)
            print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_size}/{total_size} bytes downloaded", end='', flush=True)

        print("\nDownload complete.")
        
        # Save the BytesIO content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(model_content.getvalue())
            temp_file_path = temp_file.name

        # Load the model from the temporary file
        loaded_model = load_model(temp_file_path)
        print("Model loaded successfully.")

        # Save the model to a permanent file
        save_model(loaded_model, model_filename)

        # Clean up the temporary file
        os.remove(temp_file_path)
    else:
        print(f"\nFailed to download the model. Status code: {response.status_code}")

# Check if the model file exists locally
if os.path.exists(model_filename):
    # Load the model from the saved file
    loaded_model = load_model(model_filename)
    print("Model loaded successfully.")
else:
    # Download and load the model if the file doesn't exist
    download_model()


import pickle
import requests
from io import BytesIO

# Replace 'source_tokenizer_url' and 'target_tokenizer_url' with the URLs for the respective files
source_tokenizer_url = 'https://drive.google.com/uc?id=10Dwnf4jzQkMR_QsJYbh6qHFg-GN1s9oI'
target_tokenizer_url = 'https://drive.google.com/uc?id=1Zn9UWITDiuVwzoZjN3obDCwNoALM0-wB'

# Function to load tokenizer from URL
def load_tokenizer_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        tokenizer_content = BytesIO(response.content)
        tokenizer = pickle.load(tokenizer_content)
        return tokenizer
    else:
        print(f"Failed to download the tokenizer. Status code: {response.status_code}")
        return None

# Load source tokenizer
source_tokenizer = load_tokenizer_from_url(source_tokenizer_url)
if source_tokenizer:
    print("Source tokenizer loaded successfully.")

# Load target tokenizer
target_tokenizer = load_tokenizer_from_url(target_tokenizer_url)
if target_tokenizer:
    print("Target tokenizer loaded successfully.")


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
def predict_sequence(input_text, main_model, source_tokenizer, target_tokenizer, max_target_length):
    # Tokenize the input sequence
    input_seq = source_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, padding='post')

    # Initialize the target sequence with the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']

    # Generate the output sequence word by word
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # Predict the next word in the sequence
        output_tokens = main_model.predict([input_seq, target_seq])
        
        # Sample a token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
# Sample a token with the highest probability or choose the unknown token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = target_tokenizer.index_word.get(sampled_token_index, '<unknown>')

        # Break the loop if the maximum length is reached or the end token is predicted
        if sampled_token == '<end>' or len(decoded_sentence.split()) > max_target_length:
            stop_condition = True
        else:
            decoded_sentence += sampled_token + ' '

        # Update the target sequence for the next iteration
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.strip()
# Example usage


from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/process_data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        input_text = data.get('input_string', '')  # Assuming 'input_string' is a key in your JSON payload
        # Your processing logic here to generate 'decoded_sentence'
        
        # For now, let's just use a placeholder value
        decoded_sentence = f"Processed input: {input_text}"
        
        print("Final Decoded sentence:", decoded_sentence)
        return jsonify({'result_string': decoded_sentence})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

