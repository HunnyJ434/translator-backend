# import tensorflow.keras as keras
# # Load the model
# loaded_model = keras.models.load_model('trained_model.h5')

# # Display the model summary

# import pickle

# # Load source_tokenizer
# with open('source_tokenizer.pkl', 'rb') as file:
#     source_tokenizer = pickle.load(file)

# # Load target_tokenizer
# with open('target_tokenizer.pkl', 'rb') as file:
#     target_tokenizer = pickle.load(file)
# print(target_tokenizer)

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

