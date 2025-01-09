import cv2
from torchvision.transforms.functional import to_pil_image
import torch
import base64
from io import BytesIO
from PIL import Image

def extract_frames(video_path, frame_rate=1):
    """Extract frames from a video at a given frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []  # To track the indices of selected frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Select frames based on frame rate
        if frame_count % frame_rate == 0:
            frames.append(frame)
            frame_indices.append(frame_count)

        frame_count += 1

    cap.release()
    return frames, frame_indices

def predict_frames(frames, model, transform, device):
    """Predict fake/real status for each frame."""
    predictions = []
    i = 0
    for frame in frames:
        # Convert OpenCV frame (numpy array) to PIL Image
        img = to_pil_image(frame)
        img_transformed = transform(img).unsqueeze(0).to(device)
        i = i+1
        print('frame',i,'is passed')
        
        with torch.no_grad():
            pred = model(img_transformed).flatten().tolist()
            predictions.append(pred[0])

    return predictions

def summarize_predictions(predictions):
    """Summarize frame predictions into a final video-level label."""
    fake_count = sum(1 for p in predictions if p == 1)

    if fake_count > 0:
        return "Fake Video"
    else:
        return "Real Video"

def encode_frames_as_base64(frames):
    """Encode frames (numpy arrays) as base64 strings."""
    encoded_frames = []
    for frame in frames:
        # Convert OpenCV frame (numpy array) to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save the image to a buffer
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_frames.append(encoded_image)

    return encoded_frames
