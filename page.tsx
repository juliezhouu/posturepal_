"use client";

import { useState, useEffect, useRef } from "react";
import WebcamFeed from "./components/WebcamFeed";
import StatusDisplay from "./components/StatusDisplay";
import "./globals.css"; // Import the global CSS
import { checkPosture } from "./lib/api";

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
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "20px",
      width: "90%",
      // maxWidth: "1000px",
      margin: "50px auto", // Centers the div horizontally and sets top margin
      backgroundColor: "#ffffff",
      borderRadius: "12px",
      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
    }}>
      <h1>Posture Pal</h1>
      <WebcamFeed onCapture={handleCapture} />

      <StatusDisplay 
        posture={posture} 
        message={message}
        calibrating={calibrating}
        secondsLeft={secondsLeft}
        samplesCollected={samplesCollected}
      />


    </div>
  );
}
