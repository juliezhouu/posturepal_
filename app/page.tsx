"use client";

import { useState, useEffect, useRef } from "react";
import WebcamFeed from "../components/WebcamFeed";
import StatusDisplay from "../components/StatusDisplay";
import "./globals.css"; // Import the global CSS
import { checkPosture } from "../lib/api";

export default function Home() {
  const [messages, setMessages] = useState(null);

  useEffect(() => {
    // Connect to Python backend WebSocket
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => console.log("Connected to backend");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages(data);
    };
    ws.onclose = () => console.log("Disconnected from backend");

    // Clean up connection on component unmount
    return () => ws.close();
  }, []);

  useEffect(() => {
    // Global handlers to capture unhandled promise rejections and errors.
    const onUnhandledRejection = (e: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', e.reason, e);
      // show a compact message in console with stack if available
      if (e.reason && e.reason.stack) console.error(e.reason.stack);
    };

    const onError = (event: ErrorEvent) => {
      console.error('Window error:', event.message, event.filename, event.lineno, event.colno, event.error);
      if (event.error && event.error.stack) console.error(event.error.stack);
    };

    window.addEventListener('unhandledrejection', onUnhandledRejection);
    window.addEventListener('error', onError);

    return () => {
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
      window.removeEventListener('error', onError);
    };
  }, []);

  const [posture, setPosture] = useState<"good" | "bad" | "unknown">("unknown");
  const [message, setMessage] = useState("");
  const [calibrating, setCalibrating] = useState(true);
  const [secondsLeft, setSecondsLeft] = useState<number | undefined>(undefined);
  const [samplesCollected, setSamplesCollected] = useState<number | undefined>(undefined);
  const inFlight = useRef(false);

  const handleCapture = async (image: string) => {
    if (inFlight.current) return; // skip if a request is already pending
    inFlight.current = true;
    try {
      console.log("Sending image to checkPosture...");
      const data = await checkPosture(image);
      // console.log("checkPosture returned:", databp);
      // defensive checks
      if (data && (data.posture === "good" || data.posture === "bad" || data.posture === "unknown")) {
        setPosture(data.posture);
      } else {
        console.warn("Unexpected posture value from backend, falling back to 'unknown'", data);
        setPosture("unknown");
      }
      setMessage(typeof data?.message === "string" ? data.message : "");
      setCalibrating(data?.calibrating ?? false);
      setSecondsLeft(data?.secondsLeft);
      setSamplesCollected(data?.samplesCollected);
    } catch (err) {
      console.error("checkPosture failed", err);
      setPosture("unknown");
      setMessage("Failed to contact posture service.");
    } finally {
      inFlight.current = false;
    }
  };

  return (
    <div style={{
      maxWidth: "900px",
      margin: "0 auto",
    }}>
      {/* Header */}
      <header style={{
        textAlign: "center",
        marginBottom: "1.5rem",
      }}>
        <h1 style={{
          fontSize: "2rem",
          fontWeight: "700",
          color: "#1e293b",
          marginBottom: "0.25rem",
          textShadow: "0 1px 2px rgba(0,0,0,0.05)",
        }}>PosturePal</h1>
        <p style={{
          fontSize: "0.95rem",
          color: "#64748b",
          fontWeight: "400",
        }}>Your AI-Powered Posture Assistant</p>
      </header>

      {/* Main Content Card */}
      <div style={{
        backgroundColor: "var(--bg-primary)",
        borderRadius: "16px",
        boxShadow: "var(--shadow-lg)",
        padding: "1.75rem",
        backdropFilter: "blur(10px)",
      }}>
        <div style={{
          display: "grid",
          gap: "1.5rem",
          gridTemplateColumns: "1fr",
        }}>
          <WebcamFeed onCapture={handleCapture} />
          <StatusDisplay 
            posture={posture} 
            message={message}
            calibrating={calibrating}
            secondsLeft={secondsLeft}
            samplesCollected={samplesCollected}
          />
        </div>
      </div>

      {/* Footer */}
      <footer style={{
        textAlign: "center",
        marginTop: "1.5rem",
        color: "#64748b",
        fontSize: "0.85rem",
      }}>
        <p>Monitoring your posture for better health and productivity</p>
      </footer>
    </div>
  );
}
