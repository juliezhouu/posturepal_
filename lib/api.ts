export type PostureValue = "good" | "bad" | "unknown";

export interface PostureResult {
  posture: PostureValue;
  message: string;
  calibrating?: boolean;
  secondsLeft?: number;
  samplesCollected?: number;
}

function isPostureValue(v: any): v is PostureValue {
  return v === "good" || v === "bad" || v === "unknown";
}

// Track if we're in calibration mode
let calibrationMode = true;  // Start true, will flip to false after calibration complete

export async function checkPosture(image: string): Promise<PostureResult> {
  // Preferred: send the captured image to the Python backend analyze endpoint
  try {
    console.log("Sending request to backend...");
    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        image,
        calibration_mode: calibrationMode 
      })
    });

    console.log("Response status:", res.status, res.statusText);
    
    if (!res.ok) throw new Error(`Bad response: ${res.status}`);
    const data: any = await res.json();
    
    console.log("Backend response:", data);

    // Map backend response to our PostureResult shape
    // Backend now returns: { posture: "good/bad/unknown", message: "...", calibrating: bool }
    let posture: PostureValue = "unknown";
    
    // Check if backend uses new format (has "posture" field)
    if (data?.posture) {
      // New format
      posture = data.posture === "good" ? "good" : 
                data.posture === "bad" ? "bad" : "unknown";
    } else if (data?.state) {
      // Old format compatibility
      if (data.state === "GOOD") posture = "good";
      else if (data.state === "POOR") posture = "bad";
      else if (data.state === "MODERATE" || data.state === "SIDE_TILT") posture = "bad";
      else if (data.state === "NO_POSE" || data.state === "CALIBRATING") posture = "unknown";
    }

    const message = data?.message || data?.tip || "No message from server.";
    
    // Check if calibration is complete
    if (calibrationMode && !data?.calibrating) {
      console.log("Calibration complete!");
      calibrationMode = false;
    }

    return { 
      posture, 
      message,
      calibrating: data?.calibrating,
      secondsLeft: data?.seconds_left,
      samplesCollected: data?.samples_collected
    };
  } catch (err) {
    console.warn("Analyze failed, falling back to GET /latest or mock:", err);

    // Try GET /latest as fallback
    try {
      const res2 = await fetch("http://127.0.0.1:8000/latest", { cache: "no-store" });
      if (res2.ok) {
        const data: any = await res2.json();
        let posture: PostureValue = "unknown";
        if (data?.state === "GOOD") posture = "good";
        else if (data?.state === "POOR") posture = "bad";
        else if (data?.state === "MODERATE" || data?.state === "SIDE_TILT") posture = "bad";
        const message = typeof data?.tip === "string" ? data.tip : (typeof data?.message === "string" ? data.message : "No message from server.");
        return { posture, message };
      }
    } catch (e) {
      console.warn("Fallback GET /latest also failed:", e);
    }

    // --- Final fallback mock for local testing ---
    const messages = {
      good: [
        "Great posture! Keep it up!",
        "You're sitting perfectly - nice work!",
        "Excellent posture maintained!",
      ],
      bad: [
        "Your shoulders are rounded — try sitting back.",
        "You're leaning forward too much - adjust your position.",
        "Try to align your back with your chair.",
        "Remember to keep your screen at eye level.",
      ]
    };
    const isGood = Math.random() > 0.7;
    const postureMsgs = messages[isGood ? 'good' : 'bad'];
    const randomMsg = postureMsgs[Math.floor(Math.random() * postureMsgs.length)];
    return { posture: isGood ? 'good' : 'bad', message: randomMsg };
  }
}
