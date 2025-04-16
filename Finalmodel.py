import torch
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from torchvision import models
import logging
import gradio as gr
import sys
import os
import numpy as np
from torchvision import transforms

from pymongo import MongoClient
from datetime import datetime
from playsound import playsound

client = MongoClient("mongodb://localhost:27017/")
db = client["pothole_detection"]
collection = db["alerts"]


# Assuming you've cloned the Monodepth2 repository
repo_path = 'monodepth2'  # Update with the actual path to the cloned Monodepth2 repository
sys.path.insert(0, repo_path)

import networks
from utils import download_model_if_doesnt_exist
from monodepth2.layers import disp_to_depth


# Severity thresholds (in meters, adjust as needed)
T0 = 0.5
T1 = 2.7  # Low to Medium threshold
T2 = 3.5  # Medium to High threshold

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Purpose: Loads a pre-trained Faster R-CNN model for pothole detection, configures the architecture to detect two classes (background and pothole), and loads the saved model weights.
# Parameters:
#   - model_path: Path to the saved model weights (state dictionary).
#   - device: Device (CPU or GPU) to load the model on.
# - Output:
#   - Returns the configured and loaded Faster R-CNN model.
# - Functionality: 
#   - Creates Faster R-CNN model.
#   - Reconfigures the model’s classifier head.
#   - Loads the saved weights into the model.
#   - Moves the model to the specified device and sets it to evaluation mode.

def load_detection_model(model_path, device):
    # Define the model architecture with 2 classes (background and pothole)
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
    logger.warning("Warning: Arguments other than a weight enum or `None` for 'weights' are deprecated.")

    # Replace the box predictor to have 2 classes (background, pothole)
    num_classes = 2  # 1 class for pothole + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the saved state_dict into the model
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "pothole_trainedmodel.pth")

    state_dict = torch.load(model_path, map_location=device)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    model.eval()

    return model

# - Purpose: Loads a pre-trained depth estimation model, specifically from the Monodepth2 repository, including an encoder and decoder part for predicting depth maps from images.
# - Parameters:
#   - model_name: Predefined name of the model to load.
#   - device: Device (CPU or GPU) to load the model on.
# - Output:
#   - Returns the depth encoder and depth decoder models.
# - Functionality:
#   - Downloads the model if it doesn't exist.
#   - Loads the encoder and decoder models with the respective weights.
#   - Moves the models to the specified device and sets them to evaluation mode.

def load_depth_model(model_name, device):
    # Download model if it doesn't exist
    download_model_if_doesnt_exist(model_name)

    # Path to model weights
    model_path = os.path.join("models", model_name)

    # Load the depth model
    depth_encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

    # Load the encoder weights
    encoder_path = os.path.join(model_path, "encoder.pth")
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_encoder.to(device)
    depth_encoder.eval()

    # Load the decoder weights
    decoder_path = os.path.join(model_path, "depth.pth")
    loaded_dict_dec = torch.load(decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_decoder.to(device)
    depth_decoder.eval()

    return depth_encoder, depth_decoder

# Function to predict depth from depth model
# - Purpose: Predicts the depth map for an input image using the depth encoder and decoder models.
# - Parameters:
#   - encoder: Pretrained depth encoder model.
#   - decoder: Pretrained depth decoder model.
#   - image: PIL image input.
#   - device: Device (CPU or GPU) to perform computations.
# - Output:
#   - Returns the predicted depth map as a numpy array.
# - Functionality:
#   - Resizes input image and converts it to a tensor.
#   - Moves the input tensor to the device.
#   - Runs inference through the encoder and decoder models.
#   - Resizes the output depth map to the original image size.

def predict_depth(encoder, decoder, image, device):
    # Prepare the image
    input_image = image.convert('RGB')
    original_width, original_height = input_image.size
    feed_width = 1024
    feed_height = 320

    # Width and height must be multiples of 32
    input_image_resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    input_image_transform = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    # Move the input tensor to the specified device
    input_image_transform = input_image_transform.to(device)

    # Predict depth
    with torch.no_grad():
        features = encoder(input_image_transform)
        outputs = decoder(features)

    # Resize the predicted depth to the original size of the input image
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
    scaled_disp, _ = disp_to_depth(disp_resized, 0.1, 100)
    depth_map = scaled_disp.squeeze().cpu().numpy()

    return depth_map

# Function to classify the severity of potholes based on depth
# - Purpose: Classifies the severity of potholes based on the average depth value from the predicted depth map.
# - Parameters:
#   - depth_map: Depth map of the image as a numpy array.
#   - threshold1: Threshold to classify between low and medium severity.
#   - threshold2: Threshold to classify between medium and high severity.
# - Output:
#   - Returns the severity level as a string ("Low", "Medium", or "High").
# - Functionality:
#   - Computes the average depth from the depth map.
#   - Classifies severity based on the given thresholds.

def classify_severity(depth_map, threshold0, threshold1, threshold2):
    avg_depth = np.mean(depth_map)

    if avg_depth < threshold0:
        severity = "No Severity"
    elif threshold0 <= avg_depth < threshold1:
        severity = "Low"
    elif threshold1 <= avg_depth < threshold2:
        severity = "Medium"
    else:
        severity = "High"

    print(f"Average Depth: {avg_depth:.2f} m -> Severity: {severity}")
    return severity


# Function to visualize images with bounding boxes and severity
# - Purpose: Draws bounding boxes around detected potholes on the input image and labels them with the severity of the pothole.
# - Parameters:
#   - image: PIL image input.
#   - boxes: Bounding boxes around detected potholes.
#   - labels: Labels for the bounding boxes, usually class labels.
#   - severity: Optional severity level to annotate on the image.
# - Output:
#   - Returns the image with drawn bounding boxes and labels.
# - Functionality:
#   - Draws bounding boxes and labels on the image.
#   - Converts the image to tensor format for drawing and back to PIL format for returning.

def visualize_prediction(image, boxes, labels, severity=None):
    if boxes.numel() == 0:
        logger.warning(f"No bounding boxes detected.")
        return image

    # Convert the numeric labels (tensor) to string labels
    str_labels = [f'Pothole: {severity}' if severity else 'Pothole' for label in labels]

    # Draw bounding boxes on the image
    image = F.to_tensor(image)  # Convert PIL image to tensor
    image_with_boxes = draw_bounding_boxes(image, boxes, labels=str_labels, colors="red", width=2)

    # Convert back to PIL image for returning
    image_with_boxes = F.to_pil_image(image_with_boxes)

    return image_with_boxes

# Function to run inference on an image and filter based on box size and confidence score
# - Purpose: Runs inference on an input image for pothole detection and depth estimation, filters the results based on area and confidence thresholds.
# - Parameters:
#   - detection_model: Pretrained Faster R-CNN model for object detection.
#   - depth_encoder: Pretrained depth encoder model.
#   - depth_decoder: Pretrained depth decoder model.
#   - image: PIL image input.
#   - device: Device (CPU or GPU) to perform computations.
#   - area_threshold: Minimum area threshold for bounding boxes.
#   - confidence_threshold: Minimum confidence score threshold for bounding boxes.
# - Output:
#   - Returns a dictionary containing filtered bounding boxes, labels, depth map, and severity level.
# - Functionality:
#   - Converts image to tensor and moves it to device.
#   - Runs inference using the detection model.
#   - Filters bounding boxes based on area and confidence thresholds.
#   - Predicts the depth map for the image.
#   - Classifies the severity of detected potholes based on depth.

def predict(detection_model, depth_encoder, depth_decoder, image, device, area_threshold=1000, confidence_threshold=0.5):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = detection_model(image_tensor)[0]

    filtered_boxes = []
    for box, score in zip(prediction['boxes'], prediction['scores']):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold and score > confidence_threshold:
            filtered_boxes.append(box)

    # 🔁 If NO potholes found, return immediately
    if not filtered_boxes:
        return {
            'boxes': torch.tensor([]),
            'labels': torch.tensor([]),
            'depth_map': None,
            'severity': "No potholes detected"
        }

    # 🔁 Else, process combined bounding box
    filtered_boxes_tensor = torch.stack(filtered_boxes)  # shape: [N, 4]
    x1 = torch.min(filtered_boxes_tensor[:, 0])
    y1 = torch.min(filtered_boxes_tensor[:, 1])
    x2 = torch.max(filtered_boxes_tensor[:, 2])
    y2 = torch.max(filtered_boxes_tensor[:, 3])
    big_box = torch.tensor([[x1, y1, x2, y2]])

    # Predict depth map
    depth_map = predict_depth(depth_encoder, depth_decoder, image, device)

    # Classify severity ONLY if potholes exist
    severity = classify_severity(depth_map, T0, T1, T2)

    # ✅ Store Low/Medium/High severity to MongoDB
    if severity.lower() != "no potholes detected":
        timestamp = datetime.now()
        alert_data = {
            "severity": severity,
            "timestamp": timestamp,
            "source": "image_upload"
        }
        collection.insert_one(alert_data)
        print(f"✅ Saved to MongoDB: {severity} severity at {timestamp}")

    return {
        'boxes': big_box,
        'labels': torch.tensor([1]),  # Assume 1 = pothole
        'depth_map': depth_map,
        'severity': severity
    }


# Combined Gradio interface function
# - Purpose: Combined function for the Gradio interface that processes an input image to detect potholes, predict depth, classify severity, and visualize results.
# - Parameters:
#   - image: PIL image input.
# - Output:
#   - Returns the annotated image with bounding boxes and the severity level.
# - Functionality:
#   - Detects potholes and predicts depth.
#   - Classifies severity based on predicted depth.
#   - Draws bounding boxes and labels on the image to visualize the results.


def process_image(image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    detection_model = load_detection_model("pothole_trainedmodel.pth", device)
    depth_encoder, depth_decoder = load_depth_model("mono_1024x320", device)

    if image.mode != "RGB":
        image = image.convert("RGB")

    prediction = predict(detection_model, depth_encoder, depth_decoder, image, device)
    pred_boxes = prediction['boxes']
    pred_labels = prediction['labels']
    severity = prediction['severity']

    # If no potholes
    if severity == "No potholes detected":
        return image, severity, "<div style='margin-top:10px; color:gray;'>ℹ️ No potholes detected.</div>"

    result_image = visualize_prediction(image, pred_boxes, pred_labels, severity)

    # 🎯 Top page alert (not inside severity box)
    if severity == "High":
        alert_html = """
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: red; color: white; padding: 10px 20px; border-radius: 10px; 
                    font-size: 16px; z-index: 9999; box-shadow: 0 0 10px black;">
            🚨 High Severity Detected: Not safe to drive!
        </div>
        <audio autoplay>
            <source src="file="C:\\Users\\durgadevi\\Downloads\\PotholeDetection (1)\\FinalProject_6-2-2025\\alert.mp3" type="audio/mpeg">
        </audio>
        """
    elif severity == "Medium":
        alert_html = """
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: orange; color: black; padding: 10px 20px; border-radius: 10px; 
                    font-size: 16px; z-index: 9999; box-shadow: 0 0 10px gray;">
            ⚠️ Medium Severity: Drive with caution.
        </div>
        """
    elif severity == "Low":
        alert_html = """
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: green; color: white; padding: 10px 20px; border-radius: 10px; 
                    font-size: 16px; z-index: 9999; box-shadow: 0 0 10px black;">
            ✅ Low Severity: Safe to drive.
        </div>
        """
    else:
        alert_html = ""

    if severity == "High":
        try:
            playsound(r"C:\Users\durgadevi\Downloads\PotholeDetection (1)\FinalProject_6-2-2025\alert.mp3")  # File must be in the same folder
        except Exception as e:
            print("⚠️ Couldn't play sound:", e)
    return result_image, severity, alert_html

# Main function
# - Purpose: Defines and launches the Gradio interface for the application.
# - Functionality:
#   - Sets up a Gradio interface with specified input and output formats.
#   - The interface accepts an image as input, processes it to detect potholes, classifies severity, and visualizes the results.
#   - The output includes the image with bounding boxes and a textbox for the severity level.

def main():
# Gradio interface
#     Purpose: Sets up a user interface using Gradio to allow users to upload images and get the detection and severity classification results.
# - Functionality:
#   - The fn parameter specifies the function (`process_image`) that will process the input.
#   - The inputs parameter specifies that the interface takes an image file as input.
#   - The outputs parameter specifies that the interface returns an image with bounding boxes and a text box showing the severity level.
#   - The title and description provide context to the user about the functionality of the interface.
# - Launch: Executes the interface and makes it available for user interaction.

    gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Pothole Detection Result"),
        gr.Textbox(label="Severity Level"),           # 👈 "High", "Medium", etc.
        gr.HTML(label="Alert Message")                # 👈 🚨 Popup-style alert
    ],
    title="Pothole Detection & Alert System",
    description="Upload a road image to detect potholes and get safety alerts."
).launch()


if __name__ == '__main__':
    main()