# Sign Language Recognition System v3.0 - Quick Start Guide

## What's New in v3.0?

✨ **Seven Major Improvements:**
1. ✅ **Distance-Invariant Landmarks** - Works at any camera distance
2. ✅ **Gesture-Based Recording** - No keyboard needed for recognition
3. ✅ **Stability-Based Prediction** - Repeatable and reliable sign recognition
4. ✅ **Sentence Buffer** - Accumulates words into full sentences
5. ✅ **Enhanced Visual Feedback** - See confidence, recording progress, hand status
6. ✅ **Idle State Handling** - No false predictions when hands not visible
7. ✅ **Production-Ready Code** - Clean, safe, well-commented changes

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Install requirements
pip install -r requirements.txt
```

### Quick Start
```bash
# Start the system
python main.py

# Wait for system to initialize (takes 2-3 seconds)
```

---

## How to Use (NEW GESTURE-BASED CONTROL)

### 🎯 Recognize Mode (Default)

This is the default mode - **NO keyboard needed for signing!**

**Steps:**
1. **Show your hand with palm open** 📂
   - System automatically starts recording
   - You'll see: "🎥 RECORDING..." with frame counter
   - You'll also see gesture instruction: "✊ Make a Fist to Stop"

2. **Perform your sign**
   - Keep your hand visible
   - Make the gesture/sign clearly
   - System collects up to 45 frames (~1.5 seconds)

3. **Stop by making a fist** ✊ OR remove your hand
   - Fist gesture automatically stops recording
   - Hand disappearing also stops recording
   - System recognizes the sign

4. **Hear the result** 🔊
   - Voice speaks the recognized word
   - Word is added to the sentence buffer
   - Sentence displays at bottom of screen
   - Example: "Sentence: hello thanks goodbye"

5. **Repeat** 🔁
   - Show open palm again for next sign
   - Same cycle repeats

### 📝 Record New Sign (RECORD Mode)

**To switch modes:**
```
Press 'm' → Switches to RECORD mode (red indicator)
```

**To record a new sign:**
1. **Press 'n'** - Enter the sign name you want to record
2. **Show open palm** 📂 - Recording starts automatically
3. **Make the gesture** - Perform the sign clearly
4. **Make a fist** ✊ - Stops recording and saves

**Done!** Your new sign is now in the database.

### 🗑️ Clear Sentence Buffer
```
Press 'c' → Clears the sentence buffer
            Start a new sentence
```

### 🚪 Quit
```
Press 'q' → Closes the application gracefully
```

---

## On-Screen Indicators Explained

### Top-Left Corner
```
MODE: RECOGNIZE  ← Current mode (RED = Record, GREEN = Recognize)
🖐️  Open Palm to Record  ← What gesture to make next
🟢 HANDS VISIBLE ← Hand detection status (GREEN = visible, RED = not visible)
🎥 RECORDING... (23/45 frames)  ← Recording progress
[████████░░░░░░░░░░] ← Visual progress bar
```

### Recording Progress Bar
```
[████████████░░░░░░]  
 Fills up as you record (0-45 frames)
```

### Confidence Bar
```
Confidence: 87.5%
[████████░░░░░░]  
 GREEN = High confidence (≥0.8)
 YELLOW = Medium (0.5-0.8)
 RED = Low (<0.5)
```

### Bottom of Screen
```
Sentence: hello thanks goodbye
         (Green background = confirmed words)
```

---

## Tips for Best Results

### Recording Signs
- ✅ **Keep hands in frame** - Full hand must be visible
- ✅ **Clear gestures** - Make distinct hand shapes
- ✅ **Steady motion** - Don't move too fast or too slow
- ✅ **Good lighting** - MediaPipe needs good lighting for hand detection
- ✅ **Normal distance** - Stand 1-2 feet from camera

### Recognition
- ✅ **Consistency** - Record multiple versions of same sign during setup
- ✅ **Clear separation** - Distinct hand shapes for each sign
- ✅ **Real sentences** - Signs are recognized in realistic context
- ✅ **Wait for confirmation** - System confirms with voice before adding to sentence

### Camera Setup
- 📷 **HD or higher** - 720p minimum recommended
- 💡 **Good lighting** - Well-lit room (avoid backlighting)
- 📏 **Optimal distance** - 1-3 feet from camera
- 🎭 **Plain background** - Minimal clutter helps hand detection

---

## System Architecture

### How Gesture Detection Works
```
Input: Hand landmarks from MediaPipe
   ↓
Check distance from wrist to finger tips
   ↓
Open Palm?      Fist?        No Hand?
(distance > 0.15) (distance < 0.10)
   ↓              ↓              ↓
START REC    STOP REC      STOP REC
```

### How Recognition Works
```
User performs sign (open palm → gesture → fist)
         ↓
Collect 45 frames (or stop on fist/hand removal)
         ↓
Normalize landmarks (wrist-based, scale-invariant)
         ↓
Extract hand shape features (angles between joints)
         ↓
Compare with all reference signs using DTW distance
         ↓
Check confidence (normalized distance score)
         ↓
If stable & confident → ✓ Confirm and speak
If uncertain → X Reject, wait for next sign
```

### How Sentence Building Works
```
Sign 1 recognized → ✅ Add to buffer → "hello"
         ↓
Sign 2 recognized → ✅ Add to buffer → "hello thanks"
         ↓
Sign 3 recognized → ✅ Add to buffer → "hello thanks goodbye"
         ↓
Press 'c' → Clear buffer → Ready for new sentence
```

---

## Troubleshooting

### "🔴 NO HANDS" constantly shows
**Solution:**
- Check lighting (MediaPipe needs good light)
- Move closer to camera (1-2 feet)
- Reduce background clutter
- Check camera permissions

### Signs aren't being recognized
**Solution:**
- Record more reference examples of each sign
- Ensure consistent gestures during recording
- Check hand visibility indicator (should be 🟢)
- Verify DTW distance is below threshold in console

### Voice not playing
**Solution:**
- Check system volume
- Verify pyttsx3 is installed: `pip install pyttsx3`
- Test: `python -c "import pyttsx3; pyttsx3.init().say('test').runAndWait()"`

### Recording won't stop
**Solution:**
- Make clear fist gesture
- Or move hand out of frame
- If stuck, press 'q' to quit

### Gesture detection not working
**Solution:**
- Open palm: extend ALL fingers, distance from wrist important
- Fist: close ALL fingers tightly
- Move slowly and clearly

---

## Configuration

### Adjustable Parameters

In `sign_recorder.py`, modify:
```python
class SignRecorder:
    def __init__(self, ...):
        self.seq_len = 50              # Frames per training sample
        self.dtw_threshold = 2000      # Distance threshold for recognition
        self.confirmation_threshold = 10      # Frames to confirm (NEW)
        self.confidence_threshold = 0.8       # Min confidence (NEW)
        self.sign_cooldown = 1.5       # Seconds before repeat (NEW)
        self.max_recording_frames = 45 # Auto-stop after this many (NEW)
```

### Gesture Detection Thresholds

In `sign_recorder.py`, modify:
```python
def detect_open_palm(self, results):
    avg_distance = ...
    return avg_distance > 0.15    # ← Adjust this value

def detect_fist(self, results):
    avg_distance = ...
    return avg_distance < 0.10    # ← Adjust this value
```

---

## FAQ

**Q: Do I need to press a key to record?**
A: No! Just show an open palm and it starts automatically.

**Q: Can I record the same sign twice?**
A: Yes! After 1.5 seconds cooldown, the same sign can be recognized again.

**Q: What if I make a mistake while signing?**
A: Just press 'q' to quit, or press 'c' to clear the sentence and start over.

**Q: How many signs can I record?**
A: Unlimited! Each sign is stored as reference data on disk.

**Q: Can I use this with left hand, right hand, or both?**
A: Yes! The system detects and normalizes both hands independently.

**Q: What languages are supported?**
A: Any sign language! The system learns whatever you teach it.

**Q: How accurate is the recognition?**
A: Depends on training data quality. Best with 5+ reference examples per sign.

**Q: Can I export the sentence?**
A: Currently displayed on screen and spoken. Future version can add text export.

**Q: Does it work in dark lighting?**
A: No. MediaPipe needs good lighting for reliable hand detection.

---

## Advanced Usage

### Building a Sign Dictionary

1. **Start in RECORD mode** (press 'm')
2. **For each new sign:**
   - Press 'n'
   - Enter sign name: "hello"
   - Record the sign 5+ times (press 'r' to record each, fist to stop)
   - Variation helps recognition
3. **Switch to RECOGNIZE mode** (press 'm')
4. **Test each sign** - System now recognizes them

### Batch Recording Session

```
Press 'm'        → RECORD mode
Press 'n' + "hello"    → Record "hello" 5 times
Press 'n' + "thanks"   → Record "thanks" 5 times
Press 'n' + "goodbye"  → Record "goodbye" 5 times
Press 'm'        → RECOGNIZE mode
Test each sign   → Verify they work
```

---

## System Requirements

- **Python**: 3.8+
- **Camera**: Any USB webcam, HD recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for models and data
- **OS**: Windows, macOS, Linux

---

## Performance Tips

1. **Close other applications** - Reduces CPU load
2. **Improve lighting** - Fewer frames needed for detection
3. **Larger gestures** - Easier for MediaPipe to detect
4. **Consistent speed** - Medium speed gestures work best
5. **Multiple references** - 5+ examples improves accuracy

---

## Next Steps

1. ✅ **Start the app** - `python main.py`
2. ✅ **Record 3-5 signs** - Use RECORD mode
3. ✅ **Test recognition** - Switch to RECOGNIZE mode
4. ✅ **Build sentences** - Use gesture-based control
5. ✅ **Refine accuracy** - Record more examples as needed

---

## Getting Help

**Check console output** for:
- DTW distances (for debugging)
- Confidence scores
- Frame counts
- Error messages

**Read the main file** at `SYSTEM_IMPROVEMENTS_V3.md` for detailed technical info.

---

**Version:** 3.0
**Last Updated:** January 28, 2026
**Status:** ✅ Production Ready

**Enjoy your gesture-based sign language recognition system!** 🤟

