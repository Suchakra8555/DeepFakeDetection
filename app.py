from flask import Flask, jsonify
from controllers.controller import process_video_request

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def welcome():
    return jsonify({'message': 'Welcome to DeepFake detection API'})

# Define routes
@app.route('/process_video', methods=['POST'])
def process_video():
    return process_video_request()

if __name__ == '__main__':
    app.run(debug=True)