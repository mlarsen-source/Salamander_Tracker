"use client";

import { useEffect, useRef, type FormEvent, useState } from "react";

import {
  startStream,
  stopStream,
  getStreamMetrics,
  type StreamMetrics,
} from "@/lib/api";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<StreamMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const metricsIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll metrics while streaming
  useEffect(() => {
    if (!streaming) {
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
        metricsIntervalRef.current = null;
      }
      return;
    }

    const pollMetrics = async () => {
      try {
        const data = await getStreamMetrics();
        setMetrics(data);

        // Stop polling if stream ended
        if (!data.is_streaming) {
          setStreaming(false);
        }
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      }
    };

    // Poll immediately, then every 500ms
    pollMetrics();
    metricsIntervalRef.current = setInterval(pollMetrics, 500);

    return () => {
      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
      }
    };
  }, [streaming]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setError("Choose a video first.");
      return;
    }

    setError(null);
    setMetrics(null);

    try {
      setStreaming(true);
      const response = await startStream(file);
      setStreamUrl(response.stream_url);
    } catch (err) {
      setStreaming(false);
      setError(err instanceof Error ? err.message : "Failed to start stream.");
    }
  };

  const handleStop = async () => {
    try {
      await stopStream();
    } catch (err) {
      console.error("Failed to stop stream:", err);
    } finally {
      setStreaming(false);
      setStreamUrl(null);
      setMetrics(null);
    }
  };

  const latestDetectionCount = metrics?.detection_count_over_time.length
    ? metrics.detection_count_over_time[
        metrics.detection_count_over_time.length - 1
      ].count
    : 0;

  const trackedSalamanders = Object.entries(
    metrics?.time_on_screen_by_track_id || {}
  );

  return (
    <main>
      <h1>Salamander Tracker</h1>
      <p className="muted">
        Upload a video to run YOLO detection and tracking live.
      </p>

      <section className="panel">
        <form
          onSubmit={onSubmit}
          className="row">
          <input
            type="file"
            accept="video/*"
            disabled={streaming}
            onChange={(event) => {
              const selected = event.target.files?.[0] ?? null;
              setFile(selected);
            }}
          />
          <button
            type="submit"
            disabled={streaming || !file}>
            {streaming ? "Streaming..." : "Start Stream"}
          </button>
          {streaming && (
            <button
              type="button"
              onClick={handleStop}
              style={{ marginLeft: "8px", backgroundColor: "#d32f2f" }}>
              Stop
            </button>
          )}
        </form>
        {error ? (
          <p
            className="muted"
            style={{ color: "#ff6b6b" }}>
            {error}
          </p>
        ) : null}
      </section>

      {streamUrl && (
        <section className="panel section-top-gap">
          <h2>Live Detection Stream</h2>
          <div
            style={{
              position: "relative",
              overflow: "hidden",
              borderRadius: "8px",
            }}>
            <img
              src={streamUrl}
              alt="Live stream"
              style={{
                width: "100%",
                height: "auto",
                display: "block",
              }}
            />
          </div>

          {metrics && (
            <div className="grid section-top-gap">
              <div className="panel">
                <h3>Current Frame</h3>
                <p style={{ fontSize: "20px", fontWeight: "bold" }}>
                  {metrics.frame_count || 0}
                </p>
              </div>

              <div className="panel">
                <h3>Detections This Frame</h3>
                <p style={{ fontSize: "20px", fontWeight: "bold" }}>
                  {latestDetectionCount}
                </p>
              </div>

              <div className="panel">
                <h3>FPS</h3>
                <p style={{ fontSize: "20px", fontWeight: "bold" }}>
                  {metrics.fps.toFixed(1)}
                </p>
              </div>

              <div className="panel">
                <h3>Status</h3>
                <p style={{ fontSize: "16px" }}>
                  {metrics.is_streaming ? "🔴 Live" : "✅ Complete"}
                </p>
              </div>
            </div>
          )}

          {trackedSalamanders.length > 0 && (
            <div className="panel section-top-gap">
              <h3>Tracked Salamanders</h3>
              <div style={{ display: "grid", gap: "12px" }}>
                {trackedSalamanders.map(([trackId, timeOnScreen]) => (
                  <div
                    key={trackId}
                    style={{
                      padding: "12px",
                      backgroundColor: "#f5f5f5",
                      borderRadius: "4px",
                      display: "flex",
                      justifyContent: "space-between",
                    }}>
                    <span>Track ID: {trackId}</span>
                    <span className="muted">{timeOnScreen.toFixed(2)}s</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {metrics && metrics.detection_count_over_time.length > 0 && (
            <div className="panel section-top-gap">
              <h3>Detection Timeline</h3>
              <p className="muted">
                Total frames processed:{" "}
                {metrics.detection_count_over_time.length}
              </p>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
