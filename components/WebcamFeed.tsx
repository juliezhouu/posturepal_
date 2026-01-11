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
    <div style={{
      position: "relative",
    }}>
      <div style={{
        position: "relative",
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
        border: "2px solid var(--border)",
      }}>
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={480}
          height={360}
          videoConstraints={{
            facingMode: "user",
            width: 1280,
            height: 720
          }}
          style={{
            width: "100%",
            height: "auto",
            display: "block",
          }}
        />
        {isInitializing && (
          <div style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            color: "white",
            backdropFilter: "blur(4px)",
          }}>
            <div style={{
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              padding: "1.5rem",
              borderRadius: "12px",
              textAlign: "center",
              border: "2px solid rgba(255, 255, 255, 0.2)",
            }}>
              <div style={{
                fontSize: "3rem",
                fontWeight: "700",
                marginBottom: "0.75rem",
                color: "#10b981",
              }}>
                {countdown}
              </div>
              <p style={{
                fontSize: "1.1rem",
                marginBottom: "0.5rem",
                fontWeight: "500",
              }}>
                Get Ready!
              </p>
              <p style={{
                fontSize: "0.9rem",
                opacity: 0.9,
                maxWidth: "280px",
              }}>
                Sit upright and center yourself in the camera
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
