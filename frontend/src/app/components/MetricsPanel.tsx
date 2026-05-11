"use client";

/**
 * Dashboard cards summarizing the live stream metrics.
 *
 * Two headline numbers (current frame, salamanders currently being tracked)
 * plus a small timeline summary. All values are derived from props — this
 * component is purely presentational.
 */
import type { StreamMetrics } from "@/lib/api";

import styles from "./MetricsPanel.module.css";

type MetricsPanelProps = {
  metrics: StreamMetrics;
};

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const currentFrameNumber = metrics.frame_count || 0;
  const salamandersOnLatestFrame = getLatestDetectionCount(metrics);
  const totalFramesProcessed = metrics.detection_count_over_time.length;

  return (
    <>
      <div className={`grid section-top-gap ${styles.metricsGrid}`}>
        <div className="panel">
          <h3>Current Frame</h3>
          <p className={styles.metricValue}>{currentFrameNumber}</p>
        </div>

        <div className="panel">
          <h3>Salamanders Currently Being Tracked</h3>
          <p className={styles.metricValue}>{salamandersOnLatestFrame}</p>
        </div>
      </div>

      {totalFramesProcessed > 0 && (
        <div className={`panel ${styles.timelinePanel}`}>
          <h3>Detection Timeline</h3>
          <p className={styles.timelineSummary}>
            Total frames processed: {totalFramesProcessed}
          </p>
        </div>
      )}
    </>
  );
}

/** Pull the salamander count from the most recent frame, defaulting to 0. */
function getLatestDetectionCount(metrics: StreamMetrics): number {
  const series = metrics.detection_count_over_time;
  if (series.length === 0) {
    return 0;
  }
  return series[series.length - 1].count;
}
