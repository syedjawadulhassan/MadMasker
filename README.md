# 😷 MadMasker

Real-time Face Mask Detection using **TensorFlow**, **Keras**, **OpenCV**, and **Streamlit**. This project leverages **MobileNetV2** to accurately classify whether a person is wearing a face mask from images and live webcam video.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Overview

MadMasker is a deep learning-based computer vision application that detects whether a person is wearing a face mask. The project includes:

- 🖼️ Image-based face mask detection
- 📹 Real-time webcam detection
- 🌐 Interactive Streamlit web application
- 🧠 Deep learning model using MobileNetV2
- ⚡ Fast and lightweight inference

---

## ✨ Features

- Detect masks in static images
- Real-time webcam detection
- Streamlit web interface
- Deep Learning with TensorFlow & Keras
- Lightweight MobileNetV2 architecture
- Easy to train with custom datasets

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Model Building |
| OpenCV | Computer Vision |
| Streamlit | Web Interface |
| MobileNetV2 | Image Classification |

---

## 📂 Project Structure

```text
MadMasker/
│
├── __pycache__/                # Python cache
├── css/                        # Streamlit styles
├── dataset/                    # Dataset
├── face_detector/              # Face detector model
├── images/                     # Sample images
├── incep_v3_mask_model/        # InceptionV3 model
├── Logo/                       # Project logo
├── Readme_images/              # README assets
├── ResNet50_v2/                # ResNet50V2 model
│
├── .gitignore
├── _config.yml
├── app.py                      # Streamlit app
├── detect_mask_image.py        # Image detection
├── detect_mask_video.py        # Webcam detection
├── train_mask_detector.py      # Train model
├── model2onnx.py               # Convert model to ONNX
├── search.py                   # Utility script
├── mask_detector.model         # Trained model
├── plot.png                    # Training plot
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/syedjawadulhassan/MadMasker.git
cd MadMasker
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate

**Windows**

```bash
env\Scripts\activate
```

**Linux / macOS**

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Detect Face Mask in an Image

```bash
python detect_mask_image.py --image images/test.jpg
```

### Real-Time Webcam Detection

```bash
python detect_mask_video.py
```

Press **Q** to quit.

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| Base Model | MobileNetV2 |
| Framework | TensorFlow/Keras |
| Task | Face Mask Classification |
| Classes | Mask / No Mask |
| Input | RGB Images |

---

## 📈 Future Improvements

- Face recognition integration
- YOLO-based detection
- ONNX optimization
- Docker deployment
- Cloud deployment
- Multi-face tracking

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Syed Jawad Ul Hassan**

🎓 B.Tech Computer Science & Engineering

- GitHub: https://github.com/syedjawadulhassan
- LinkedIn: https://linkedin.com/in/syed-jawad-ul-hassan

If you found this project useful, consider giving it a ⭐ on GitHub.