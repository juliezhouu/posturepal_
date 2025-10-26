"use client";

import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";

interface Props {
  onCapture: (image: string) => void;
}

export default function WebcamFeed({ onCapture }: Props) {
  const webcamRef = useRef<Webcam>(null);
  const [lastImage, setLastImage] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number>(10);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);

  const capture = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) onCapture(imageSrc);
  };

  const captureAndStore = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setLastImage(imageSrc);
      onCapture(imageSrc);
    }
  };

  // Initial countdown effect
  useEffect(() => {
    if (isInitializing && countdown > 0) {
      const timer = setInterval(() => {
        setCountdown(prev => prev - 1);
      }, 1000);

      return () => clearInterval(timer);
    } else if (countdown === 0 && isInitializing) {
      setIsInitializing(false);
      // Take an immediate capture when initialization completes so the UI responds right away
      capture();
    }
  }, [countdown, isInitializing]);

  // Regular capture interval
  useEffect(() => {
    if (!isInitializing) {
      // Capture every 1 second
      const interval = setInterval(() => {
        capture();
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isInitializing, onCapture]);

  return (
    <div>
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={600}
        height={450}
        videoConstraints={{
          facingMode: "user",
          width: 1280,
          height: 720
        }}
        style={{
          borderRadius: "12px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)"
        }}
      />
      {isInitializing && (
        <div style={{
          marginTop: "10px",
          textAlign: "center",
          padding: "10px",
          backgroundColor: "#f0f8ff",
          borderRadius: "8px",
          border: "1px solid #4CAF50"
        }}>
          <p style={{ margin: "0", fontSize: "1.1rem", color: "#2E7D32" }}>
            Please sit upright and center yourself in the camera
          </p>
          <p style={{ margin: "4px 0 0 0", fontSize: "1.2rem", fontWeight: "bold", color: "#1B5E20" }}>
            Starting in {countdown} seconds...
          </p>
        </div>
      )}
    </div>
  );
}
