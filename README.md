# Ojibwe Translation API 🌐📖

A **Flask-based API** that serves an **Ojibwe to English translation model**, trained using **60,000+ data entries**.  
The API is deployed on **Google Cloud App Engine**, providing real-time translations through HTTP endpoints.

## 🚀 Features
- 🧠 **Machine learning-powered translation** using a trained TensorFlow model.
- 🔥 **Flask API** for easy integration with web and mobile applications.
- ☁️ **Deployed on Google Cloud App Engine** for scalability and reliability.
- 🛠 **RESTful API design**, supporting **GET** and **POST** requests.

## 🛠 Technologies Used
- **Python** (Flask, TensorFlow)
- **Google Cloud App Engine**
- **REST API**
- **JSON for data exchange**

## 📌 API Endpoints

### **1️⃣ Translate Ojibwe to English**

## Usage
git clone https://github.com/HunnyJ434/ojibwe-translation-api.git
cd ojibwe-translation-api

pip install -r requirements.txt
python app.py

### Deploying through gcloud
gcloud auth login
gcloud config set project [PROJECT_ID]
gcloud app deploy


📝 License
This project is licensed under the MIT License.


