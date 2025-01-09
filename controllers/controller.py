from flask import request, jsonify
from models.model import load_model
from utils.video_processing import (
    extract_frames,
    predict_frames,
    summarize_predictions,
    encode_frames_as_base64
)
import os
import torchvision.transforms as transforms
import torch

# Define the transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the required input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_video_request():
    """Controller function to process a video request."""
    video_file = request.files.get('video')
    model_type = request.form.get('model_type', 'linear')  # Default to linear if not provided
    frame_rate = int(request.form.get('frame_rate', 1))    # Default frame rate of 1

    if not video_file:
        return jsonify({'error': 'No video file provided'}), 400

    # Save video to a temporary file
    video_path = f"temp_video_{os.getpid()}.mp4"
    video_file.save(video_path)

    try:
        # Load the chosen model
        model = load_model(model_type)

        # Extract frames
        frames, frame_indices = extract_frames(video_path, frame_rate)

        # Predict frame-by-frame
        predictions = predict_frames(frames, model, transform, device)

        # Summarize predictions
        video_label = summarize_predictions(predictions)

        # Identify fake frames
        fake_frame_indices = [frame_indices[i] for i, p in enumerate(predictions) if p == 1]
        fake_frames = [frames[i] for i, p in enumerate(predictions) if p == 1]

        # Encode fake frames as base64
        fake_frames_base64 = encode_frames_as_base64(fake_frames)

        # Return the result
        return jsonify({
            'video_label': video_label,
            'fake_frame_indices': fake_frame_indices,
            'fake_frames': fake_frames_base64
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary video
        if os.path.exists(video_path):
            os.remove(video_path)
