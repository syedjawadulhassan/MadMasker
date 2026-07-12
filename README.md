# 😷 MadMasker

<div align="center">

### Real-Time Face Mask Detection using Deep Learning

Built with **TensorFlow**, **Keras**, **OpenCV**, and **Streamlit** using the **MobileNetV2** architecture.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</div>

---

## 📖 Overview

MadMasker is a real-time face mask detection application powered by Deep Learning and Computer Vision. It uses the MobileNetV2 architecture to classify whether a person is wearing a face mask from both static images and live webcam streams.

The project includes an interactive Streamlit interface, making it simple to test face mask detection directly from a web browser.

---

## ✨ Features

- 😷 Face mask detection from images
- 🎥 Real-time webcam detection
- 🌐 Interactive Streamlit web application
- 🧠 MobileNetV2 deep learning model
- ⚡ Fast and lightweight inference
- 📱 Simple and user-friendly interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Model Development |
| OpenCV | Computer Vision |
| Streamlit | Web Application |
| MobileNetV2 | Image Classification Model |

---

## 📂 Project Structure

```text
MadMasker/
│
├── css/
├── face_detector/
├── images/
├── Logo/
├── Readme_images/
│
├── .gitignore
├── app.py
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/syedjawadulhassan/MadMasker.git

cd MadMasker
```

---

### 2️⃣ Create a Virtual Environment

```bash
python -m venv env
```

---

### 3️⃣ Activate the Virtual Environment

**Windows**

```bash
env\Scripts\activate
```

**Linux / macOS**

```bash
source env/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Detect Face Mask from an Image

```bash
python detect_mask_image.py --image images/test.jpg
```

---

### Real-Time Webcam Detection

```bash
python detect_mask_video.py
```

Press **Q** to quit.

---

### Run Streamlit Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| Base Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Task | Binary Image Classification |
| Classes | Mask / No Mask |
| Input Size | 224 × 224 RGB |

---

## 📸 Application Preview

> Add screenshots inside the `Readme_images` folder and replace the filenames below.

### Home Page

```markdown
![Home](Readme_images/home.png)
```

### Image Detection

```markdown
![Image Detection](Readme_images/image_detection.png)
```

### Webcam Detection

```markdown
![Webcam Detection](Readme_images/webcam.png)
```

---

## 📈 Future Improvements

- YOLO-based face detection
- ONNX model optimization
- Docker support
- Cloud deployment
- Face recognition integration
- Multi-person tracking
- Performance optimization

---

## 🤝 Contributing

Contributions are welcome.

1. Fork this repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is released under the **MIT License** and is intended for educational, research, and demonstration purposes.

---

## 👨‍💻 Author

### Syed Jawad Ul Hassan

**B.Tech Computer Science & Engineering**

GitHub:
https://github.com/syedjawadulhassan

LinkedIn:
https://linkedin.com/in/syed-jawad-ul-hassan

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and supports future development.

---

<div align="center">

Made with ❤️ by **Syed Jawad Ul Hassan**

</div>
