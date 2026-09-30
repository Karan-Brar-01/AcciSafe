# Impact-Net (AcciSafe)

Impact-Net is an Edge-AI Traffic Crash Triage & Severity Assessment system designed for Emergency Operations Centers (EOC). It provides a command-and-control dashboard for edge crash triage, analyzing video feeds in real-time to detect traffic collisions and assess their severity using computer vision and physics-based heuristics.

## Features

- **Emergency Operations Center Dashboard**: A fully featured, dark-themed Streamlit UI for monitoring and triage (`app.py`).
- **Real-Time Video Analysis**: Uses YOLOv8 for accurate vehicle detection and tracking (`pipeline.py`).
- **Kinetic Severity Index (KSI)**: Calculates collision severity using dense optical flow physics, estimating $\Delta v$ (change in velocity) and impact angles.
- **Automated Dispatch Telemetry**: Generates structured JSON payloads ready for 911/EMS computer-aided dispatch (CAD) handoff.
- **Debris Plume Detection**: Identifies high-energy airborne dust/smoke clouds for critical severity escalation.
- **Sample Generation**: Includes a script to download real-world crash clips or synthesize dashboard-style animations (`download_samples.py`).

## Project Structure

- `app.py`: The Streamlit dashboard application and UI presentation layer.
- `pipeline.py`: The core computer vision engine, handling YOLO inference, optical flow, physics calculations, and telemetry generation.
- `download_samples.py`: Utility script to download or synthesize sample video clips for testing.
- `requirements.txt`: Python package dependencies.
- `packages.txt`: System-level dependencies (e.g., ffmpeg).
- `samples/`: Directory containing pre-loaded sample videos (populated by `download_samples.py`).
- `outputs/`: Directory where annotated video outputs are saved.

## Installation

### 1. System Dependencies
Ensure you have the required system libraries installed (especially for OpenCV and video processing). On Debian/Ubuntu:

```bash
sudo apt-get update
xargs -a packages.txt sudo apt-get install -y
```

*(Note: `packages.txt` includes `ffmpeg`, `libsm6`, and `libxext6`)*

### 2. Python Dependencies
It is recommended to use a virtual environment. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Sample Videos
Before running the dashboard, populate the `samples/` directory with test clips. You can download real clips or generate synthetic ones by running:

```bash
python download_samples.py
```

### 2. Launch the Dashboard
Start the Streamlit application:

```bash
streamlit run app.py
```

### 3. Using the Dashboard
1. Open the provided local URL (usually `http://localhost:8501`) in your web browser.
2. In the **Mission Control** sidebar, select a **Detection model** (e.g., YOLOv8n-Traffic for CPU edge triage).
3. Set the **Sensitivity threshold** (Low, Medium, High).
4. Select a **Pre-loaded video** from the dropdown or upload your own custom feed (`.mp4`, `.avi`, `.mov`).
5. Click **▶ Run Analysis**.
6. View the real-time AI-annotated feed, telemetry charts, and the generated Emergency Dispatch Payload.

## How It Works

Impact-Net uses a multi-stage pipeline for crash detection:
1. **Detection & Tracking**: YOLOv8 detects vehicles (cars, trucks, buses, motorcycles, etc.) frame-by-frame.
2. **Motion Estimation**: Bounding box center displacement and dense optical flow (Gunnar-Farneback) estimate the velocity of each vehicle.
3. **Anomaly & Collision Detection**: The system monitors for sudden decelerations ($\Delta v$ spikes), interacting bounding boxes, and scene-wide motion bursts (e.g., debris plumes).
4. **Severity Scoring (KSI)**: Based on the estimated change in velocity and impact angle, the Kinetic Severity Index is calculated, mapping to a severity level (1 to 5).
5. **Dispatch Recommendation**: Generates actionable insights (e.g., "Monitor incident" vs. "IMMEDIATE trauma dispatch") based on the severity level.
