# 🕳️ Pothole Detection System using Deep Learning

This project is a real-time pothole detection system using deep learning and computer vision. It identifies potholes in images or video frames and provides alerts through a web interface. Detection results are stored in a MongoDB database, and users receive real-time notifications via sound alerts.

---

## 🚀 Features

- 🔍 Detects potholes using a pretrained **Faster R-CNN with ResNet-50** model.
- 🖼️ Draws **bounding boxes** around detected potholes.
- 💽 Stores detection logs with timestamps in **MongoDB**.
- 🔊 Plays alert sound on detection.
- 🌐 Simple and interactive **Gradio UI** for testing.
- 📦 Can be extended for real-time camera or mobile use.

---

## 🧠 Model Used

- **Faster R-CNN** (Region-based Convolutional Neural Network)
- **Backbone**: ResNet-50
- **Dataset**: [COCO (Common Objects in Context)](https://cocodataset.org/)

---

## 🛠️ Tech Stack

| Component      | Technology         |
|----------------|--------------------|
| Framework      | PyTorch, TorchVision |
| Web Interface  | Gradio             |
| Backend Server | Flask (Optional)   |
| Database       | MongoDB            |
| Alerts         | playsound (Python) |

---

## ⚙️ How It Works

1. User uploads an image through the Gradio interface.
2. The model processes the image and detects potholes.
3. Detected regions are highlighted using bounding boxes.
4. Detection results are stored in MongoDB.
5. An alert sound is played if potholes are detected.

---

## 🧪 Future Improvements

- Add real-time camera feed and GPS tracking.
- Classify severity of potholes.
- Deploy in smart vehicles or city surveillance systems.
- Replace local database with cloud MongoDB Atlas or Firebase.

---

 
## 📁 Project Structure

```bash
├── app.py              # Main application file  
├── model/              # Pretrained model setup  
├── utils/              # Utility functions  
├── templates/          # HTML templates (if using Flask)  
├── requirements.txt    # Required packages  
└── README.md           # Project documentation
```
---

## 📦 Installation

```bash
git clone https://github.com/yourusername/pothole-detection
cd pothole-detection
pip install -r requirements.txt
python app.py



