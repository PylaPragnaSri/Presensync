from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
from detector import load_model, process_first_n_seconds, PROOF_DIR

app = Flask(__name__)
CCTV_VIDEO_PATH = "cctv_input.mp4"

# load model once at startup
MODEL = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_video", methods=["POST"])
def process_video():
    faculty_count_raw = request.form.get('faculty_count', None)
    tolerance_raw = request.form.get('tolerance', "1")

    try:
        faculty_count = int(faculty_count_raw) if faculty_count_raw else None
        tolerance = int(tolerance_raw)
    except ValueError:
        return jsonify({"error": "faculty_count and tolerance must be integers"}), 400

    # Run detection using the preloaded MODEL
    result = process_first_n_seconds(
        CCTV_VIDEO_PATH,
        model=MODEL,
        seconds=5,
        subsample_every_n_frames=1,
        save_proof_dir=PROOF_DIR,
        start_time_sec=0.0
    )

    auto_count = int(result["proof_metadata"].get("detected_persons_in_proof_frame", 0))

    decision = None
    if faculty_count is not None:
        decision = "ACCEPTED" if abs(auto_count - faculty_count) <= tolerance else "REJECTED"

    proof_path = result.get("proof_image_path")
    proof_url = None
    if proof_path:
        proof_filename = os.path.basename(proof_path)
        proof_url = url_for('proof_file', filename=proof_filename)

    return jsonify({
        "decision": decision,
        "proof_url": proof_url
    })

@app.route("/proof/<path:filename>")
def proof_file(filename):
    return send_from_directory(PROOF_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
