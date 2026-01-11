"use client";

import { useEffect, useRef, useState } from 'react';

interface Props {
  posture: "good" | "bad" | "unknown";
  message: string;
  calibrating?: boolean;
  secondsLeft?: number;
  samplesCollected?: number;
}

export default function StatusDisplay({ posture, message, calibrating, secondsLeft, samplesCollected }: Props) {
  // Use null as initial state to indicate we haven't checked permissions yet
  const [notifPermission, setNotifPermission] = useState<NotificationPermission | null>(null);
  // Track if we're in browser environment
  const [isBrowser, setIsBrowser] = useState(false);

  // Set isBrowser on mount
  useEffect(() => {
    setIsBrowser(true);
  }, []);

  // Helper function to safely create a notification
  const createNotification = async (title: string, options?: NotificationOptions) => {
    if (!('Notification' in window)) {
      console.log('Notifications not supported');
      return;
    }
    
    try {
      // Create and wait for the notification to be shown
      const notification = new Notification(title, options);
      
      // Return a promise that resolves when the notification is closed
      return new Promise((resolve) => {
        notification.onshow = () => {
          console.log('Notification shown:', title);
          resolve(true);
        };
        notification.onerror = (e) => {
          console.error('Notification error:', e);
          resolve(false);
        };
      });
    } catch (e) {
      console.error('Failed to create notification:', e);
      return false;
    }
  };

  // Check notification permission on mount, but only in browser
  useEffect(() => {
    if (!isBrowser) return;
    
    const checkPermission = async () => {
      if (!('Notification' in window)) return;
      
      try {
        // Get the current permission state
        const permission = Notification.permission;
        console.log('Current notification permission:', permission);
        setNotifPermission(permission);
      } catch (e) {
        console.error('Failed to check notification permission:', e);
        setNotifPermission(null);
      }
    };

    checkPermission();
  }, [isBrowser]);

  // Track last time we showed a posture notification (ms since epoch)
  const lastNotifiedRef = useRef<number | null>(null);

  // Show notification only for bad posture
  useEffect(() => {
    if (!isBrowser) return;
    if (!message || posture === "unknown") return;
    
    console.log('Notification check:', { posture, message, permission: Notification.permission });
    
    // Only show notification if posture is bad
    if (posture !== "bad") {
      console.log('Notification skipped - posture is:', posture);
      return;
    }
    
    const showNotification = async () => {
      const now = Date.now();
      // If we've recently shown a notification within the last 10s, skip
      if (lastNotifiedRef.current && now - lastNotifiedRef.current < 10000) {
        console.log('Skipping notification - last shown', now - (lastNotifiedRef.current || 0), 'ms ago');
        return;
      }

      console.log('Attempting to show notification...');
      if ('Notification' in window && Notification.permission === "granted") {
        await createNotification(`⚠️ Posture Alert`, {
          body: `${message}\n(${new Date().toLocaleTimeString()})`,
          requireInteraction: false,
          tag: String(now),  // Unique tag for each notification
          silent: false            // Allow sound
        });

        lastNotifiedRef.current = now;
        console.log('Showing notification for posture:', posture, 'with message:', message);
      } else {
        console.log('Notification not shown - permission:', Notification.permission);
      }
    };

    showNotification();
  }, [message, posture, isBrowser]);

  // Request permission with proper async handling
  const requestPermission = async () => {
    if (!isBrowser || !('Notification' in window)) {
      console.log('Notifications not supported');
      return;
    }
    
    try {
      console.log('Requesting notification permission...');
      const result = await Notification.requestPermission();
      console.log('Permission request result:', result);
      setNotifPermission(result);
      
      if (result === 'granted') {
        // Send a test notification
        await createNotification('Notifications Enabled', {
          body: 'You will now receive posture check notifications!'
        });
      }
    } catch (e) {
      console.error('Permission request failed:', e);
      setNotifPermission(null);
    }
  };

  // Only render notification UI if we're in the browser
  if (!isBrowser) {
    return (
      <div
        style={{
          padding: "2rem",
          borderRadius: "16px",
          textAlign: "center",
        }}
      >
        <p style={{ fontSize: "1.2rem" }}>{message}</p>
      </div>
    );
  }

  const getPostureColor = () => {
    if (calibrating) return "var(--primary)";
    switch (posture) {
      case "good": return "var(--secondary)";
      case "bad": return "var(--danger)";
      default: return "var(--text-secondary)";
    }
  };

  const getPostureIcon = () => {
    if (calibrating) return "⏱️";
    switch (posture) {
      case "good": return "✅";
      case "bad": return "⚠️";
      default: return "❓";
    }
  };

  return (
    <div
      style={{
        padding: "1.5rem",
        borderRadius: "12px",
        backgroundColor: "var(--bg-secondary)",
        border: "2px solid var(--border)",
      }}
    >
      {calibrating ? (
        <div style={{ textAlign: "center" }}>
          <div style={{
            fontSize: "2.5rem",
            marginBottom: "0.75rem",
          }}>
            {getPostureIcon()}
          </div>
          <h3 style={{
            fontSize: "1.25rem",
            marginBottom: "0.5rem",
            color: getPostureColor(),
            fontWeight: "600",
          }}>
            Calibrating Your Posture
          </h3>
          {secondsLeft !== undefined && (
            <p style={{
              fontSize: "1rem",
              color: "var(--text-secondary)",
              marginBottom: "1rem",
            }}>
              Time remaining: <strong style={{ color: "var(--primary)" }}>{secondsLeft}s</strong>
            </p>
          )}
          {samplesCollected !== undefined && (
            <div style={{ marginTop: "0.75rem" }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "0.5rem",
              }}>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Progress</span>
                <span style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--primary)" }}>
                  {samplesCollected}/10
                </span>
              </div>
              <div style={{
                width: "100%",
                height: "10px",
                backgroundColor: "#e2e8f0",
                borderRadius: "5px",
                overflow: "hidden",
              }}>
                <div style={{
                  width: `${Math.min((samplesCollected / 10) * 100, 100)}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%)",
                  borderRadius: "5px",
                  transition: "width 0.4s ease-out",
                }} />
              </div>
            </div>
          )}
          <div style={{
            marginTop: "1rem",
            padding: "0.75rem",
            backgroundColor: "rgba(37, 99, 235, 0.1)",
            borderRadius: "10px",
            border: "1px solid rgba(37, 99, 235, 0.2)",
          }}>
            <p style={{
              fontSize: "0.875rem",
              color: "var(--text-primary)",
              margin: 0,
              lineHeight: "1.5",
            }}>
              💡 Maintain a comfortable, upright posture while looking at your screen
            </p>
          </div>
        </div>
      ) : (
        <div style={{ textAlign: "center" }}>
          <div style={{
            fontSize: "3rem",
            marginBottom: "0.75rem",
          }}>
            {getPostureIcon()}
          </div>
          <div style={{
            fontSize: "1.1rem",
            color: "var(--text-primary)",
            fontWeight: "500",
            lineHeight: "1.6",
          }}>
            {message}
          </div>
          {posture !== "unknown" && (
            <div style={{
              marginTop: "1rem",
              padding: "0.75rem 1.25rem",
              backgroundColor: posture === "good" ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)",
              borderRadius: "10px",
              border: `2px solid ${posture === "good" ? "var(--secondary)" : "var(--danger)"}`,
            }}>
              <span style={{
                fontSize: "1rem",
                fontWeight: "600",
                color: getPostureColor(),
              }}>
                {posture === "good" ? "Great Job!" : "Needs Attention"}
              </span>
            </div>
          )}
        </div>
      )}
      {'Notification' in window && notifPermission !== 'granted' && (
        <div style={{
          marginTop: "1.5rem",
          paddingTop: "1.5rem",
          borderTop: "1px solid var(--border)",
        }}>
          <button
            onClick={requestPermission}
            style={{
              padding: "0.625rem 1.5rem",
              borderRadius: "10px",
              backgroundColor: "var(--primary)",
              color: "white",
              border: "none",
              cursor: "pointer",
              fontSize: "0.9rem",
              fontWeight: "600",
              boxShadow: "var(--shadow)",
            }}
          >
            🔔 Enable Notifications
          </button>
          <div style={{
            marginTop: "0.625rem",
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
          }}>
            Get alerts when your posture needs correction
          </div>
        </div>
      )}
    </div>
  );
}
