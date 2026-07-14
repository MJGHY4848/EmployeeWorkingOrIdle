# Employee Work Monitoring System

A computer vision-based employee monitoring system that uses **YOLOv8 Pose Estimation** to detect employee posture and estimate work activity in real time. The system identifies whether an employee is sitting, standing, or inactive, tracks working duration, and stores work logs for further analysis.

---

## Features

- Real-time human pose estimation using YOLOv8
- Posture classification
  - Standing
  - Sitting
  - Unclear
- Hand movement detection
- Employee work time tracking
- Automatic CSV work log generation
- Live confidence score display
- FPS monitoring
- Multi-person detection

---

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- CSV
- Datetime

---

## Project Structure

```
Employee-Work-Monitoring/
│
├── plain.py              # Main application
├── best.pt               # Trained YOLOv8 pose model
├── work_log.csv          # Generated work logs
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Update the video path inside `plain.py` if needed.

Run the application

```bash
python plain.py
```

Press **Q** to quit.

---

## Output

The application displays

- Bounding boxes
- Employee ID
- Posture
- Work status
- Confidence score
- FPS
- Total detected employees

It also generates

```
work_log.csv
```

containing

- Person ID
- Start Time
- End Time
- Total Work Time

---

## Model

This project uses a custom-trained **YOLOv8 Pose Estimation** model.

Replace `best.pt` with your own trained model if required.

---

## Future Improvements

- BoT-SORT or ByteTrack integration for persistent tracking
- Re-identification (ReID)
- Heatmap visualization
- Dashboard for work analytics
- Activity recognition beyond posture estimation
- Database integration

---

## License

This project is intended for educational and research purposes.