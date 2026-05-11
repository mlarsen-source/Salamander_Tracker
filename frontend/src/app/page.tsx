"use client";

/**
 * Salamander Tracker home page.
 *
 * Orchestrates three child components:
 *   1. `UploadForm`    — pick a video and start/stop the live stream
 *   2. `LiveStream`    — render the MJPEG feed once a stream URL exists
 *   3. `MetricsPanel`  — show the latest stats while the stream is running
 *
 * The polling logic lives in the `useStreamMetrics` hook so this file only
 * has to coordinate state transitions.
 */
import { useState } from "react";

import { LiveStream } from "./components/LiveStream";
import { MetricsPanel } from "./components/MetricsPanel";
import { UploadForm } from "./components/UploadForm";
import { startStream, stopStream } from "@/lib/api";
import { useStreamMetrics } from "@/lib/useStreamMetrics";

export default function HomePage() {
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const { metrics, isStreaming, reset: resetMetrics } = useStreamMetrics(streamUrl);

  /** Upload the file, then surface the MJPEG URL so the stream + polling start. */
  const handleStartStream = async (videoFile: File) => {
    setErrorMessage(null);
    setStreamUrl(null);
    resetMetrics();

    try {
      setIsStarting(true);
      const startResponse = await startStream(videoFile);
      setStreamUrl(startResponse.stream_url);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to start stream.",
      );
    } finally {
      setIsStarting(false);
    }
  };

  /** Tell the backend to halt and clear all local stream state. */
  const handleStopStream = async () => {
    try {
      await stopStream();
    } catch (error) {
      console.error("Failed to stop stream:", error);
    } finally {
      setStreamUrl(null);
      resetMetrics();
    }
  };

  return (
    <main>
      <h1>Salamander Tracker</h1>
      <p className="muted">
        Upload a video to run YOLO detection and tracking live.
      </p>

      <section className="panel">
        <UploadForm
          isStarting={isStarting}
          isStreaming={isStreaming}
          errorMessage={errorMessage}
          onStart={handleStartStream}
          onStop={handleStopStream}
        />
      </section>

      {streamUrl && (
        <section className="panel section-top-gap">
          <h2>Live Detection Stream</h2>
          <LiveStream streamUrl={streamUrl} />
          {metrics && <MetricsPanel metrics={metrics} />}
        </section>
      )}
    </main>
  );
}
