Presensync – An automated attendance verification system


## 📖 Overview

**PresenSync** is an AI-based automated attendance system designed to monitor classroom presence using **CCTV footage**.  
The system uses **YOLOv8 (You Only Look Once)** for accurate person detection and generates **proof images** with bounding boxes.  

It processes short video clips (around 5 seconds) to determine the **number of persons detected**, compares it against the expected faculty count, and automatically provides a decision — **Accepted** or **Rejected** — along with visual proof.

---

## 🧠 Key Features

✅ Detects and counts people in CCTV footage using **YOLOv8**  
✅ Generates proof images with **bounding boxes**  
✅ Automatically validates attendance against expected counts  
✅ Works fully **offline** and **on CPU** (no GPU required)  
✅ Flask-based **web interface** for user interaction  
✅ Easy deployment using a one-click **`.bat`** script  

---

## 🧩 Technologies Used

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.11 |
| Web Framework | Flask |
| AI Model | YOLOv8n (Nano Model) |
| Libraries | OpenCV, Ultralytics, NumPy, Pillow |
| Environment | Virtualenv |
| Hardware | CPU (GPU optional) |

---

## ⚙️ Installation and Setup

### 🔧 Step 1: Clone this repository
```bash
git clone https://github.com/<yourusername>/PresenSync.git
cd PresenSync
🧱 Step 2: Create a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
📦 Step 3: Install dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Step 4: Run the application
You can either run directly or use the .bat file.

Option 1:

bash
Copy code
python app.py
Option 2 (simpler):
Double-click run_test.bat

🌐 Web Interface
Open your browser and go to:

👉 http://127.0.0.1:5000

The interface allows you to:

Input expected faculty count

Process CCTV video automatically (uses default sample)

View “Accepted” / “Rejected” pop-up messages

See generated proof image in /proof_output/ folder

📸 Example Output
Frame	Description
Detected 9 people accurately with bounding boxes

Output Summary Example:

json
Copy code
{
  "auto_count": 9,
  "faculty_count": 5,
  "tolerance": 1,
  "decision": "ACCEPTED",
  "proof_image": "proof_output/proof_frame_idx149_count9.png"
}
🧮 Algorithm and Model Details
Model Used: YOLOv8n (Nano version by Ultralytics)

Detection Type: Person class only

Post-processing:

Non-Maximum Suppression (NMS) to remove duplicate detections

Median filtering for robust count stability

Decision Logic:

Compares detected person count with expected faculty count

Accepts if within tolerance; rejects otherwise

🏗️ Project Structure
graphql
Copy code
PresenSync/
│
├── app.py                 # Flask main application
├── detector.py            # YOLOv8 detection logic
├── requirements.txt       # Required Python packages
├── run_test.bat           # One-click launcher
│
├── templates/             # HTML files for UI
│   └── index.html
│
├── static/                # CSS/JS or frontend assets
├── proof_output/          # Stores proof images with bounding boxes
└── cctv_input.mp4         # Sample CCTV input video
🚀 Future Enhancements
✨ Integrate face recognition for individual identification
✨ Real-time CCTV stream processing
✨ Database integration for attendance logs
✨ Cloud dashboard for multi-class monitoring
✨ Model fine-tuning to improve accuracy under occlusions

🧾 License
This project is released under the MIT License.
Feel free to use, modify, and distribute it for educational or research purposes.

🙌 Acknowledgments
Ultralytics YOLOv8 for real-time detection

OpenCV for frame processing

Flask for the lightweight web interface

NumPy for fast numerical computations

💬 Contact
Author: Om
Project: PresenSync
GitHub: https://github.com/<yourusername>/PresenSync

🧠 “PresenSync – Automating attendance, one frame at a time.”

yaml
Copy code

---

✅ **How to use it:**
1. Open Notepad or VS Code.  
2. Copy–paste all the text above.  
3. Save it as `README.md` (exact name, no `.txt`).  
4. Place it inside your project folder.  
5. Run:
   ```bash
   git add README.md

   git commit -m "Added professional README"
   git push origin main
