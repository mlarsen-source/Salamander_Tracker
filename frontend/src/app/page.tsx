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
import styles from "./page.module.css";
import { startStream, stopStream } from "@/lib/api";
import { useStreamMetrics } from "@/lib/useStreamMetrics";

export default function HomePage() {
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const {
    metrics,
    isStreaming,
    reset: resetMetrics,
  } = useStreamMetrics(streamUrl);

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
        error instanceof Error ? error.message : "Failed to start stream."
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
    <main className={styles.shell}>
      <header className={styles.pageHeader}>
        <h2 className={styles.title}>Salamander Tracker</h2>
      </header>

      <div className={styles.workspace}>
        <aside className={styles.sidebar}>
          <div
            className={styles.specimenStrip}
            aria-label="Study metadata">
            <div className={styles.specimenStat}>
              <span className={styles.specimenLabel}>Model</span>
              <span className={styles.specimenValue}>YOLO tracking</span>
            </div>
            <div className={styles.specimenStat}>
              <span className={styles.specimenLabel}>Mode</span>
              <span className={styles.specimenValue}>Live inference</span>
            </div>
          </div>

          <section
            className={styles.controlPanel}
            aria-labelledby="controls-title">
            <p className={styles.panelKicker}>Acquisition</p>
            <h2
              className={styles.panelTitle}
              id="controls-title">
              Video Intake
            </h2>
            <UploadForm
              isStarting={isStarting}
              isStreaming={isStreaming}
              errorMessage={errorMessage}
              onStart={handleStartStream}
              onStop={handleStopStream}
            />
          </section>
        </aside>

        <section
          className={styles.streamPanel}
          aria-labelledby="stream-title">
          <div className={styles.panelHeader}>
            <div>
              <p className={styles.panelKicker}>Observation Feed</p>
              <h2
                className={styles.panelTitle}
                id="stream-title">
                Live Detection Stream
              </h2>
            </div>
            <div
              className={styles.statusPill}
              data-active={isStreaming}>
              <span
                className={styles.statusDot}
                aria-hidden="true"
              />
              {isStreaming ? "Streaming" : isStarting ? "Starting" : "Standby"}
            </div>
          </div>

          <div className={styles.streamBody}>
            {streamUrl ? (
              <>
                <LiveStream streamUrl={streamUrl} />
                {metrics && <MetricsPanel metrics={metrics} />}
              </>
            ) : (
              <div className={styles.emptyState}>
                <div className={styles.emptyStateInner}>
                  <p>
                    Select a video file and start the stream to view annotated
                    frames, detection counts, and tracking telemetry.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
