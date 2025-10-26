import mediapipe as mp
import cv2
import numpy as np
import math
import time
import base64
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Initialize Gemini AI if API key exists
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        # Use the correct Gemini model name
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        print(f"[AI] GEMINI_API_KEY present: {bool(API_KEY)}")
        print(f"[AI] Gemini enabled: {GEMINI_MODEL is not None}")
    except Exception as e:
        print(f"[AI] Failed to init Gemini SDK: {e}")
        GEMINI_MODEL = None
else:
    print("[AI] No GEMINI_API_KEY or GOOGLE_API_KEY found in environment. AI feedback disabled.")

# Constants
EPS = 1e-6
EMA_ALPHA = 0.5       # reacts faster (was 0.3)
GOOD_CUT = 1.0        # threshold between GOOD and MODERATE (higher = more stable, less likely to show "Quick reset")
POOR_CUT = 1.4        # threshold for POOR (higher = more stable)
SIDE_TILT_CUT = 2.5   # side-tilt gate (higher = less sensitive to side tilting)
SENS_GAIN = 1.0       # multipliers all slouch z-scores (lower = less sensitive = more stable)
CALIB_SECONDS = 10.0  # calibration period in seconds

def direction_from_sign(val: float) -> str:
    if val > 0.04: return "right"
    if val < -0.04: return "left"
    return ""

def generate_feedback_with_gemini(summary: dict) -> str | None:
    """Return ONE short sentence or None; handles empty/blocked responses."""
    if GEMINI_MODEL is None:
        return None

    prompt = (
        "Provide one brief, general posture description (max 120 chars). "
        "Summarize the user's current posture state in one short, "
        "neutral sentence without giving direct advice. Focus on describing. "
        f"Contextual data: {json.dumps(summary, ensure_ascii=False)}"
    )

    try:
        cfg = GenerationConfig(
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            max_output_tokens=128,
            response_mime_type="text/plain",
            candidate_count=1,
        )
        resp = GEMINI_MODEL.generate_content(prompt, generation_config=cfg)

        # Extract first text from response
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", []) or []):
                t = getattr(part, "text", None)
                if t and t.strip():
                    return " ".join(t.strip().split())[:160]

        # If we get here, try a simpler prompt
        resp2 = GEMINI_MODEL.generate_content(
            f"Offer a concise posture observation based on: {json.dumps(summary, ensure_ascii=False)}",
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=128,
                response_mime_type="text/plain",
                candidate_count=1,
            )
        )
        
        for cand in getattr(resp2, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", []) or []):
                t = getattr(part, "text", None)
                if t and t.strip():
                    return " ".join(t.strip().split())[:160]

        return None

    except Exception as e:
        print(f"[AI] Gemini error (handled): {e}")
        return None

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ai_test")
def ai_test():
    summary = {
        "trigger": "TEST",
        "now_state": "POOR",
        "persist_s": 5.0,
        "direction": "right",
        "zscores": {
            "head_forward": 1.3,
            "neck_flex": 0.9,
            "torso_forward": 0.8,
            "chin_to_chest": 1.6,
            "side_tilt": 0.4
        }
    }
    msg = generate_feedback_with_gemini(summary)

    return {
        "using_ai": GEMINI_MODEL is not None,
        "model": getattr(GEMINI_MODEL, "model_name", None) if GEMINI_MODEL else None,
        "msg": msg,
    }

# Pydantic model for request validation
class ImagePayload(BaseModel):
    image: str
    calibration_mode: bool = False

# Global state
browser_calib = {
    "delta": [], "head_fwd": [], "torso_deg": [], "neck_deg": [],
    "shoulder_ratio": [], "ear_over_hip": [], "interseg_deg": [],
    "shoulder_line_deg": [], "hip_line_deg": [], "lat_shift": [],
    "chin_chest": [],
}
browser_calib_start = None
browser_calib_done = False
browser_base = {}  # Will hold mu/sd after calibration

# Helper functions
def safe_std(x):
    if len(x) < 2: return 1.0
    s = float(np.std(x))
    return s if s > 1e-6 else 1.0

def to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def mid(p1, p2):
    return (p1 + p2) * 0.5

def dist(a, b):
    return float(np.linalg.norm(a - b))

def deg_from_vertical(base, top):
    v = top - base
    dx = float(v[0])
    dy = float(v[1])
    return math.degrees(math.atan2(abs(dx), abs(dy) + EPS))

def line_deg(p_left, p_right):
    return math.degrees(math.atan2(float(p_right[1] - p_left[1]),
                                 float(p_right[0] - p_left[0])))

def lateral_shift(sho_mid, hip_mid, torso_len):
    return abs(float(sho_mid[0] - hip_mid[0])) / max(torso_len, EPS)

def point_line_distance(p, a, b):
    """Perpendicular distance from point p to infinite line a--b (pixels)."""
    ap = p - a
    ab = b - a
    denom = np.linalg.norm(ab) + EPS
    proj = ab * (np.dot(ap, ab) / (denom**2))
    ortho = ap - proj
    return float(np.linalg.norm(ortho))

def avg_visibility(lms, ids):
    vals = [lms[i].visibility for i in ids]
    return float(np.mean(vals)) if vals else 0.0

def z_pos(x, mu, sigma):
    return max(0.0, (x - mu) / max(sigma, 1.0))

def z_neg(x, mu, sigma):
    return max(0.0, (mu - x) / max(sigma, 1.0))

def direction_from_sign(val: float) -> str:
    if val > 0.04: return "right"
    if val < -0.04: return "left"
    return ""

def make_tip(payload: dict, mu: dict, sd: dict):
    issues = []
    state = payload["state"]
    z = {}
    def Z(name, x):
        z[name] = {"pos": z_pos(x, mu.get(name, 0.0), sd.get(name, 1.0)),
                   "neg": z_neg(x, mu.get(name, 0.0), sd.get(name, 1.0))}
    
    Z("delta", payload["delta_deg"])
    Z("head_fwd", payload["head_fwd"])
    Z("neck_deg", payload["neck_deg"])
    Z("torso_deg", payload["torso_deg"])
    Z("shoulder_ratio", payload["shoulder_ratio"])
    Z("lat_shift", payload["lat_shift"])
    Z("chin_chest", payload["chin_chest"])

    sho_x = payload["_sho_x"]
    hip_x = payload["_hip_x"]
    torso_len_px = payload["_torso_len_px"] + EPS
    signed_shift = (sho_x - hip_x) / torso_len_px
    dir_word = direction_from_sign(signed_shift)

    if state == "SIDE_TILT" or z["lat_shift"]["pos"] >= 1.5:
        issues.append("Side tilt" + (f" to the {dir_word}" if dir_word else ""))

    if z["delta"]["neg"] >= 1.0: issues.append("Neck and torso aligned (slouch cue)")
    if z["head_fwd"]["pos"] >= 1.0: issues.append("Head forward of shoulders")
    if z["neck_deg"]["pos"] >= 1.0: issues.append("Neck flexed forward")
    if z["torso_deg"]["pos"] >= 1.0: issues.append("Torso leaning forward")
    if z["shoulder_ratio"]["neg"] >= 1.0: issues.append("Shoulders likely protracted")
    if z["chin_chest"]["neg"] >= 1.0: issues.append("Chin dropped toward chest")

    if state == "POOR":
        if z["head_fwd"]["pos"] >= 1.2 and z["delta"]["neg"] >= 1.2:
            tip = "Pull chin back over shoulders and lift sternum; grow tall through the crown."
        elif z["chin_chest"]["neg"] >= 1.2:
            tip = "Float the chin away from the chest and lift your sternum. Lengthen the back of your neck."
        elif z["torso_deg"]["pos"] >= 1.2:
            tip = "Hinge up from mid-back and stack shoulders over hips."
        else:
            tip = "Sit tall: ears over shoulders, shoulders over hips."
        if dir_word: tip += f" Also, center from the {dir_word}."
        return tip, issues
    if state == "SIDE_TILT":
        tip = f"Center your trunk. Shift off the {dir_word or 'tilted'} side until level."
        return tip, issues
    if state == "MODERATE":
        return "Quick reset: tuck chin slightly, raise chest, relax shoulders down/back.", issues
    if state == "CALIBRATING":
        return "Calibrating upright baseline. Sit naturally upright.", issues
    return "Nice posture. Keep chin level and shoulders relaxed.", issues

@app.post("/analyze")
def analyze_image(payload: ImagePayload):
    """Accept a base64-encoded image (data URL or raw base64) and return posture analysis.
    Handles both calibration mode and regular analysis.
    """
    global browser_calib, browser_calib_start, browser_calib_done, browser_base
    
    try:
        # Initialize calibration if needed
        if payload.calibration_mode and not browser_calib_start:
            browser_calib_start = time.time()
            browser_calib_done = False
            browser_base = {}
            for k in browser_calib:
                browser_calib[k] = []
                
        # Decode image
        b64 = payload.image.split(",", 1)[1] if "," in payload.image else payload.image
        jpg = base64.b64decode(b64)
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"state": "NO_POSE", "calibrating": not browser_calib_done, "tip": "Invalid image"}, status_code=400)

        # Process with MediaPipe Pose
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

        tick_time = time.time()

        # No pose detected
        if not results.pose_landmarks:
            seconds_left = CALIB_SECONDS - (tick_time - browser_calib_start) if browser_calib_start else None
            resp = {
                "neck_deg": None, "torso_deg": None, "delta_deg": None,
                "state": "NO_POSE",
                "visibility_ok": False,
                "tip": "Move into view and face the camera.",
                "issues": [],
                "calibrating": not browser_calib_done,
                "seconds_left": round(seconds_left, 1) if seconds_left is not None else None,
                "samples_collected": len(browser_calib["delta"]) if not browser_calib_done else None,
            }
            return JSONResponse(resp)

        # Extract landmarks and compute metrics
        lms = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        L_SH = to_xy(lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        R_SH = to_xy(lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        L_HP = to_xy(lms[mp_pose.PoseLandmark.LEFT_HIP.value], w, h)
        R_HP = to_xy(lms[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h)
        L_ER = to_xy(lms[mp_pose.PoseLandmark.LEFT_EAR.value], w, h)
        R_ER = to_xy(lms[mp_pose.PoseLandmark.RIGHT_EAR.value], w, h)
        NOSE = to_xy(lms[mp_pose.PoseLandmark.NOSE.value], w, h)

        SHO = mid(L_SH, R_SH)
        HIP = mid(L_HP, R_HP)
        EAR = mid(L_ER, R_ER)
        torso_len = dist(HIP, SHO) + EPS

        vis_ok = avg_visibility(lms, [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_EAR.value,
            mp_pose.PoseLandmark.RIGHT_EAR.value,
        ]) >= 0.6

        # Compute core metrics
        neck_deg = round(deg_from_vertical(SHO, EAR), 2)
        torso_deg = round(deg_from_vertical(HIP, SHO), 2)
        delta_deg = round(abs(neck_deg - torso_deg), 2)
        head_fwd = round(abs(float(EAR[0] - SHO[0])) / max(torso_len, EPS), 3)
        ear_over_hip = round(abs(float(EAR[0] - HIP[0])) / max(torso_len, EPS), 3)
        shoulder_width = dist(L_SH, R_SH)
        shoulder_ratio = round(shoulder_width / max(torso_len, EPS), 3)

        v1 = SHO - HIP
        v2 = EAR - SHO
        c = float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS)
        c = min(1.0, max(-1.0, c))
        interseg_deg = round(math.degrees(math.acos(c)), 2)

        shoulder_line_deg = round(line_deg(L_SH, R_SH), 2)
        hip_line_deg = round(line_deg(L_HP, R_HP), 2)
        lat_shift = round(lateral_shift(SHO, HIP, torso_len), 3)

        # Chin-to-chest
        nose_to_shoulder_px = point_line_distance(NOSE, L_SH, R_SH)
        chin_chest = round(nose_to_shoulder_px / max(torso_len, EPS), 3)  # smaller ⇒ chin tucked

        # Handle calibration
        if not browser_calib_done and vis_ok:
            # Collect calibration data
            browser_calib["delta"].append(delta_deg)
            browser_calib["head_fwd"].append(head_fwd)
            browser_calib["torso_deg"].append(torso_deg)
            browser_calib["neck_deg"].append(neck_deg)
            browser_calib["shoulder_ratio"].append(shoulder_ratio)
            browser_calib["ear_over_hip"].append(ear_over_hip)
            browser_calib["interseg_deg"].append(interseg_deg)
            browser_calib["shoulder_line_deg"].append(shoulder_line_deg)
            browser_calib["hip_line_deg"].append(hip_line_deg)
            browser_calib["lat_shift"].append(lat_shift)
            browser_calib["chin_chest"].append(chin_chest)

            # Check if calibration is complete
            if tick_time - browser_calib_start >= CALIB_SECONDS:
                browser_base.update({k: {"mu": float(np.mean(v)) if len(v) > 0 else 0.0,
                                      "sd": safe_std(v) if len(v) > 1 else 1.0}
                                  for k, v in browser_calib.items()})
                browser_calib_done = True

        # Generate response
        out = {
            "neck_deg": neck_deg,
            "torso_deg": torso_deg,
            "delta_deg": delta_deg,
            "chin_chest": chin_chest,
            "head_fwd": head_fwd,
            "ear_over_hip": ear_over_hip,
            "shoulder_ratio": shoulder_ratio,
            "interseg_deg": interseg_deg,
            "shoulder_line_deg": shoulder_line_deg,
            "hip_line_deg": hip_line_deg,
            "lat_shift": lat_shift,
            "_sho_x": float(SHO[0]),
            "_hip_x": float(HIP[0]),
            "_torso_len_px": float(dist(HIP, SHO)),
            "visibility_ok": bool(vis_ok),
            "calibrating": not browser_calib_done
        }

        # If calibration is still ongoing
        if not browser_calib_done:
            out["state"] = "CALIBRATING"
            out["seconds_left"] = round(max(0.0, CALIB_SECONDS - (tick_time - browser_calib_start)), 1)
            out["samples_collected"] = len(browser_calib["delta"])
            tip_text = "Calibrating upright baseline. Sit naturally upright."
            out["tip"] = tip_text
            out["issues"] = []
            return JSONResponse(out)

        # After calibration: compute scores
        mu = {k: browser_base[k]["mu"] for k in browser_base}
        sd = {k: browser_base[k]["sd"] for k in browser_base}

        s_delta = z_neg(delta_deg, mu["delta"], sd["delta"])
        s_head = z_pos(head_fwd, mu["head_fwd"], sd["head_fwd"])
        s_neck = z_pos(neck_deg, mu["neck_deg"], sd["neck_deg"])
        s_torso = z_pos(torso_deg, mu["torso_deg"], sd["torso_deg"])
        s_shrat = z_neg(shoulder_ratio, mu["shoulder_ratio"], sd["shoulder_ratio"])
        s_inter = z_neg(interseg_deg, mu["interseg_deg"], sd["interseg_deg"]) * 0.2
        s_chin = z_neg(chin_chest, mu["chin_chest"], sd["chin_chest"])  # smaller = worse

        # Boost sensitivity
        s_delta *= SENS_GAIN
        s_head *= SENS_GAIN
        s_neck *= SENS_GAIN
        s_torso *= SENS_GAIN
        s_shrat *= SENS_GAIN
        s_inter *= SENS_GAIN
        s_chin *= SENS_GAIN

        slouch_score = (0.35*s_delta + 0.25*s_head + 0.15*s_neck + 0.15*s_torso +
                       0.10*s_shrat + s_inter + 0.20*s_chin)

        z_sh_line = abs(shoulder_line_deg - mu["shoulder_line_deg"]) / max(sd["shoulder_line_deg"], 1.0)
        z_hip_line = abs(hip_line_deg - mu["hip_line_deg"]) / max(sd["hip_line_deg"], 1.0)
        z_lat = z_pos(lat_shift, mu["lat_shift"], sd["lat_shift"])
        side_tilt_score = 0.6*z_sh_line + 0.4*z_hip_line + 0.5*z_lat

        # Side-tilt flag
        side_flag = side_tilt_score is not None and side_tilt_score > SIDE_TILT_CUT

        # Hard triggers
        hard_poor = False
        if chin_chest <= (mu["chin_chest"] - 0.8 * sd["chin_chest"]):  # Chin very close to chest vs baseline
            hard_poor = True
        if (delta_deg <= (mu["delta"] - 1.0 * sd["delta"])) and (head_fwd >= (mu["head_fwd"] + 0.8 * sd["head_fwd"])):  # Classic slouch
            hard_poor = True
        if (delta_deg < 4.0) and (neck_deg > 12.0):  # Absolute safety net
            hard_poor = True

        # Determine state
        if side_flag and slouch_score < GOOD_CUT:
            state = "SIDE_TILT"
        elif hard_poor or slouch_score >= POOR_CUT:
            state = "POOR"
        elif slouch_score >= GOOD_CUT:
            state = "MODERATE"
        else:
            state = "GOOD"

        out["state"] = state
        out["slouch_score"] = round(slouch_score, 3)
        out["side_tilt_score"] = round(side_tilt_score, 3)
        out["params"] = {
            "good_cut": GOOD_CUT,
            "poor_cut": POOR_CUT,
            "side_tilt_cut": SIDE_TILT_CUT,
            "calib_seconds": CALIB_SECONDS,
            "sens_gain": SENS_GAIN
        }

        tip_text, issues = make_tip(out, mu, sd)
        
        # Generate AI feedback if available
        summary = {
            "trigger": "ANALYSIS",
            "now_state": state,
            "persist_s": 0.0,
            "direction": direction_from_sign((float(SHO[0]) - float(HIP[0])) / float(dist(HIP, SHO))),
            "zscores": {
                "head_forward": round(z_pos(head_fwd, mu["head_fwd"], sd["head_fwd"]), 2),
                "neck_flex": round(z_pos(neck_deg, mu["neck_deg"], sd["neck_deg"]), 2),
                "torso_forward": round(z_pos(torso_deg, mu["torso_deg"], sd["torso_deg"]), 2),
                "chin_to_chest": round(z_neg(chin_chest, mu["chin_chest"], sd["chin_chest"]), 2),
                "side_tilt": round(side_tilt_score, 2)
            }
        }

        ai_feedback = generate_feedback_with_gemini(summary) if GEMINI_MODEL else None
        message = ai_feedback or tip_text

        # Map posture states to frontend values
        posture_value = "unknown"
        if state == "GOOD":
            posture_value = "good"
        elif state in ["POOR", "MODERATE", "SIDE_TILT"]:
            posture_value = "bad"
        
        response = {
            "posture": posture_value,
            "message": message,
            "calibrating": not browser_calib_done,
            "debug": {
                "state": state,
                "issues": issues,
                "measurements": {
                    "neck_angle": round(neck_deg, 1),
                    "torso_angle": round(torso_deg, 1),
                    "delta_angle": round(delta_deg, 1),
                    "head_forward": round(head_fwd, 3),
                    "shoulder_ratio": round(shoulder_ratio, 3),
                    "chin_chest": round(chin_chest, 3)
                }
            }
        }

        return JSONResponse(response)

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
