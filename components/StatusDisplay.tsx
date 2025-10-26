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
          padding: "20px",
          borderRadius: "12px",
          marginTop: "20px",
          textAlign: "center",
        }}
      >
        <p style={{ fontSize: "1.2rem" }}>{message}</p>
      </div>
    );
  }

  return (
    <div
      style={{
        padding: "20px",
        borderRadius: "12px",
        marginTop: "20px",
        textAlign: "center",
      }}
    >
      {calibrating ? (
        <div>
          <p style={{ fontSize: "1.2rem", marginBottom: "10px" }}>Calibrating your posture...</p>
          {secondsLeft !== undefined && (
            <p style={{ fontSize: "1rem", color: "#666" }}>Time remaining: {secondsLeft}s</p>
          )}
          {samplesCollected !== undefined && (
            <div style={{ marginTop: "10px" }}>
              <p style={{ fontSize: "0.9rem", color: "#666" }}>Samples collected: {samplesCollected}</p>
              <div style={{ 
                width: "100%", 
                height: "8px", 
                backgroundColor: "#eee", 
                borderRadius: "4px",
                marginTop: "5px" 
              }}>
                <div style={{
                  width: `${Math.min((samplesCollected / 10) * 100, 100)}%`,
                  height: "100%",
                  backgroundColor: "#4CAF50",
                  borderRadius: "4px",
                  transition: "width 0.3s ease-in-out"
                }} />
              </div>
            </div>
          )}
          <p style={{ fontSize: "0.9rem", marginTop: "15px", color: "#666" }}>
            Please maintain a comfortable, upright posture while looking at your screen
          </p>
        </div>
      ) : (
        <p style={{ fontSize: "1.2rem" }}>{message}</p>
      )}
      {'Notification' in window && notifPermission !== 'granted' && (
        <div style={{ marginTop: 12 }}>
          <button 
            onClick={requestPermission} 
            style={{ 
              padding: '8px 12px', 
              borderRadius: 6,
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Enable notifications
          </button>
          <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
            Current permission: {notifPermission ?? 'unknown'}
          </div>
        </div>
      )}
    </div>
  );
}
