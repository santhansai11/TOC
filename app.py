"""
Entry point – run from project root:
    python app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from app import app, start_capture

if __name__ == "__main__":
    print("=" * 60)
    print("  🧠  EmotionAI – Real-Time Facial Emotion Detection")
    print("  Open browser: http://127.0.0.1:5000")
    print("=" * 60)
    start_capture()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
