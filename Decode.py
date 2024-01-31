#!/path/to/your/myenv/python

# # import firebase_admin
# # from firebase_admin import credentials
# # from firebase_admin import firestore
# # import csv
# # # Use a service account.
# # cred = credentials.Certificate('credentials.json')
 
# # app = firebase_admin.initialize_app(cred)

# # db = firestore.client()

  


# # db.collection("words").document("Ojib-Eng").set(dictionary)
# print(sentence_pairs)
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException
# from bs4 import BeautifulSoup
# import time
# def extract_and_append_info(soup, output_file):
#     # Find the first ul element with class name 'list-group ng-scope'
#     first_ul = soup.find('ul', class_='list-group ng-scope')

#     # If the ul element is found, proceed to extract information
#     if first_ul:
#         # Find all list items with class name 'list-group-item ng-scope' inside the first ul
#         list_items = first_ul.find_all(class_='list-group-item ng-scope')

#         # Loop through the found list items
#         for list_item in list_items:
#             # Find the strong tag with class 'ng-binding' inside each list item
#             strong_tag = list_item.find('strong', class_='ng-binding')
#             text1 = strong_tag.text.strip() if strong_tag else ""

#             # Find the span with class 'ng-scope' inside each list item
#             span_tag = list_item.find('span', class_='ng-scope')
#             text2 = span_tag.text.strip() if span_tag else ""

#             # Write the line to the file in the specified format
#             output_line = f"{text1}: {text2}\n"
#             output_file.write(output_line)

# # Replace 'your_url_here' with the actual URL of the website
# url = 'https://dictionary.nishnaabemwin.atlas-ling.ca/#/browse'
# # Initialize a Selenium WebDriver (make sure you have the appropriate driver installed)
# driver = webdriver.Chrome()

# # Open the URL in the browser
# driver.get(url)

# # Open a file in append mode
# with open('output.txt', 'a', encoding='utf-8') as output_file:
#     click_counter = 0
#     # Loop through the pages
#     while True:
#         # Use BeautifulSoup to parse the HTML content
#         soup = BeautifulSoup(driver.page_source, 'html.parser')

#         # Extract and append information
#         extract_and_append_info(soup, output_file)

#         # Click the "Next" button
#         try:
#             if click_counter == 6:
#                 next_button_xpath = f'/html/body/div[2]/div[2]/div/div/div/div/uib-accordion/div/div[1]/div[2]/div/ul[1]/li[15]/a'
#             elif click_counter >= 7:
#                 next_button_xpath = f'/html/body/div[2]/div[2]/div/div/div/div/uib-accordion/div/div[1]/div[2]/div/ul[1]/li[16]/a'
#             else:
#                 next_button_xpath = f'/html/body/div[2]/div[2]/div/div/div/div/uib-accordion/div/div[1]/div[2]/div/ul[1]/li[14]/a'

#             next_button = WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.XPATH, next_button_xpath))
#             )
#             next_button.click()
#             click_counter += 1
#         except TimeoutException:
#             # If the "Next" button is not found, break the loop
#             break
#         time.sleep(3)
# # Close the browser window
# driver.quit()


# import ccxt

# # Replace 'your_api_key' and 'your_secret_key' with your actual API key and secret key
# api_key = '8vKnN2BRIcs1zuFdb8YwqxtZkyZ+VT/aGXN6iIaqhaD2tbspR2BOWo0F'
# secret_key = 'AIxD+pIfqOw23VLVXfHs4XbAIdLW0RjY9wmKVaV0am9X4gg+l47lXSdE+PPVZLKh06Z+0Iz+ToTRPO1DrAjYag=='

# Initialize the Kraken exchange object with your API 

# 8vKnN2BRIcs1zuFdb8YwqxtZkyZ+VT/aGXN6iIaqhaD2tbspR2BOWo0F
# AIxD+pIfqOw23VLVXfHs4XbAIdLW0RjY9wmKVaV0am9X4gg+l47lXSdE+PPVZLKh06Z+0Iz+ToTRPO1DrAjYag==


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

# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# def predict_sequence(input_text, main_model, source_tokenizer, target_tokenizer, max_target_length):
#     # Tokenize the input sequence
#     input_seq = source_tokenizer.texts_to_sequences([input_text])
#     input_seq = pad_sequences(input_seq, padding='post')

#     # Initialize the target sequence with the start token
#     target_seq = np.zeros((1, 1))
#     target_seq[0, 0] = target_tokenizer.word_index['<start>']

#     # Generate the output sequence word by word
#     stop_condition = False
#     decoded_sentence = ''

#     while not stop_condition:
#         # Predict the next word in the sequence
#         output_tokens = main_model.predict([input_seq, target_seq])
        
#         # Sample a token with the highest probability
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
# # Sample a token with the highest probability or choose the unknown token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_token = target_tokenizer.index_word.get(sampled_token_index, '<unknown>')

#         # Break the loop if the maximum length is reached or the end token is predicted
#         if sampled_token == '<end>' or len(decoded_sentence.split()) > max_target_length:
#             stop_condition = True
#         else:
#             decoded_sentence += sampled_token + ' '

#         # Update the target sequence for the next iteration
#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] = sampled_token_index

#     return decoded_sentence.strip()
# Example usage


from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
@app.route('/api/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    input_text = data.get('input_string', '')
    decoded_sentence = "Hello"
    print("Final Decoded sentence:", decoded_sentence)
    # Perform some processing in your Python function
    result_string = decoded_sentence
    return jsonify({'result_string': result_string})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

