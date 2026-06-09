
# Waste Classification AI - Real-Time Predictor

This repository contains a real-time computer vision system that uses a trained machine learning model to classify different types of waste materials (such as Paper, Glass, Tin, and Organic waste) via a live webcam feed. Developed as part of a university AI mini-project.

## 📁 Project Structure

```text
📁 MINI-PROJECT-AI/
│
├── 📁 .venv/               # Local Python Virtual Environment
├── keras_Model.h5          # Trained Keras AI model file (HDF5 format)
├── labels.txt              # Class index labels mapping
├── prediction.py           # Core live webcam UI and classification script
└── requirements.txt        # System library dependencies list

```

---

## 🛠️ Setup & Installation Instructions

Follow these steps to set up your local development workspace inside the virtual environment:

### 1. Open Project Workspace

Open your terminal or command prompt inside the project root directory:

```bash
cd path/to/MINI-PROJECT-AI

```

### 2. Activate the Virtual Environment (`.venv`)

Before installing dependencies, ensure you activate the included local environment wrapper.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1

```

**Windows (Command Prompt / CMD):**

```cmd
.venv\Scripts\activate.bat

```

**Mac / Linux:**

```bash
source .venv/bin/activate

```

> 💡 **Tip:** Once activated, your terminal command line prompt will be prefixed with `(.venv)`.

### 3. Install Package Requirements

Install the core deep learning frameworks and video processing libraries:

```bash
pip install -r requirements.txt

```

*(Note: Downloading tensorflow may take a few minutes depending on your network connection speed).*

---

## 🚀 Running the Predictor

Once the dependencies are installed and you have confirmed that `keras_Model.h5` and `labels.txt` are sitting in the root folder, run the live script:

```bash
python prediction.py

```

---

## ⌨️ Controls & Troubleshooting

* **To Exit:** Press the **ESC** key on your keyboard while focusing on the active camera popup window to shut down the video stream cleanly.
* **Camera Not Found Error:** If the script fails to detect your camera or selects the wrong one (e.g., integrated laptop camera instead of an external USB webcam), open `prediction.py` and modify line 19:
```python
camera = cv2.VideoCapture(0) # Change 0 to 1 if using an external USB camera

```



---

## 📦 Dependencies Breakdown

The project relies on these strict environment package baselines as written in `requirements.txt`:

* **tensorflow**: Runs the foundational deep learning model architecture and evaluates the neural network arrays.
* **opencv-python**: Captures real-time camera frames and draws the visual diagnostic UI overlays on screen.
* **numpy**: Normalizes and shapes frame multi-dimensional matrices to prepare them for model ingestion.

```
