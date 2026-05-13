"use client";

/**
 * Dashboard cards summarizing the live stream metrics.
 *
 * Shows the required scientific readouts:
 * detection count over time and per-salamander time on screen.
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
  const peakDetectionCount = getPeakDetectionCount(metrics);
  const recentCountRows = getRecentDetectionCountRows(metrics);
  const timeOnScreenRows = getTimeOnScreenRows(metrics);
  const currentPositions = metrics.current_positions || [];
  const longestTimeOnScreen = Math.max(
    1,
    ...timeOnScreenRows.map((row) => row.seconds)
  );

  return (
    <section
      className={styles.metricsPanel}
      aria-label="Stream telemetry">
      <div className={styles.metricsGrid}>
        <div className={styles.metricTile}>
          <h3>Frame</h3>
          <p className={styles.metricValue}>{currentFrameNumber}</p>
          <span className={styles.metricUnit}>processed</span>
        </div>

        <div className={styles.metricTile}>
          <h3>Current Count</h3>
          <p className={styles.metricValue}>{salamandersOnLatestFrame}</p>
          <span className={styles.metricUnit}>detections</span>
        </div>

        <div className={styles.metricTile}>
          <h3>Peak Count</h3>
          <p className={styles.metricValue}>{peakDetectionCount}</p>
          <span className={styles.metricUnit}>in one frame</span>
        </div>
      </div>

      <div className={styles.readoutGrid}>
        <section className={styles.readoutPanel}>
          <div className={styles.readoutHeader}>
            <h3>Detection Count Over Time</h3>
            <p>
              {totalFramesProcessed > 0
                ? `${recentCountRows.length} recent time readings`
                : "Waiting for processed frames"}
            </p>
          </div>

          <div
            className={styles.countTable}
            role="table"
            aria-label="Detection count over time">
            <div
              className={styles.countTableHeader}
              role="row">
              <span>Video Time</span>
              <span>Visible</span>
            </div>
            {recentCountRows.length > 0 ? (
              recentCountRows.map((record, index) => (
                <div
                  className={styles.countTableRow}
                  data-current={index === recentCountRows.length - 1}
                  key={record.timestampSec}
                  role="row">
                  <span>{formatTimestamp(record.timestampSec)}</span>
                  <strong>
                    {record.count} salamander{record.count === 1 ? "" : "s"}
                  </strong>
                </div>
              ))
            ) : (
              <span className={styles.emptyReadout}>
                No detections recorded yet
              </span>
            )}
          </div>
        </section>

        <div className={styles.readoutStackedContainer}>
          <section className={styles.readoutPanel}>
            <div className={styles.readoutHeader}>
              <h3>Live Coordinates</h3>
              <p>
                {currentPositions.length > 0
                  ? `${currentPositions.length} visible`
                  : "No detections"}
              </p>
            </div>

            <div className={styles.positionList}>
              {currentPositions.length > 0 ? (
                currentPositions.map((pos, index) => (
                  <div
                    className={styles.positionRow}
                    key={index}>
                    <div className={styles.positionMeta}>
                      <span>
                        {pos.track_id
                          ? `ID ${pos.track_id}`
                          : `Det ${index + 1}`}
                      </span>
                      {pos.confidence && (
                        <span className={styles.confidence}>
                          {(pos.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className={styles.positionCoords}>
                      <div className={styles.coordRow}>
                        <span>x:</span>
                        <strong>{pos.x_center.toFixed(0)}</strong>
                      </div>
                      <div className={styles.coordRow}>
                        <span>y:</span>
                        <strong>{pos.y_center.toFixed(0)}</strong>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className={styles.emptyPositions}>
                  No detections on current frame
                </div>
              )}
            </div>
          </section>

          <section className={styles.readoutPanel}>
            <div className={styles.readoutHeader}>
              <h3>Time On Screen</h3>
              <p>
                {timeOnScreenRows.length > 0
                  ? `${timeOnScreenRows.length} tracked salamander${
                      timeOnScreenRows.length === 1 ? "" : "s"
                    }`
                  : "Waiting for track IDs"}
              </p>
            </div>

            <div className={styles.trackList}>
              {timeOnScreenRows.length > 0 ? (
                timeOnScreenRows.map((row) => (
                  <div
                    className={styles.trackRow}
                    key={row.trackId}>
                    <div className={styles.trackMeta}>
                      <span>Salamander {row.trackId}</span>
                      <span>{row.seconds.toFixed(1)}s</span>
                    </div>
                    <div className={styles.trackBarRail}>
                      <span
                        className={styles.trackBar}
                        style={{
                          width: `${Math.max(
                            4,
                            (row.seconds / longestTimeOnScreen) * 100
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <div className={styles.emptyTracks}>
                  No tracked salamanders yet
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}

function getLatestDetectionCount(metrics: StreamMetrics): number {
  const series = metrics.detection_count_over_time;
  if (series.length === 0) {
    return 0;
  }
  return series[series.length - 1].count;
}

function getPeakDetectionCount(metrics: StreamMetrics): number {
  return metrics.detection_count_over_time.reduce(
    (peak, record) => Math.max(peak, record.count),
    0
  );
}

function getRecentDetectionCountRows(metrics: StreamMetrics): Array<{
  timestampSec: number;
  count: number;
}> {
  const latestReadingBySecond = new Map<
    number,
    { timestampSec: number; count: number }
  >();

  for (const record of metrics.detection_count_over_time) {
    latestReadingBySecond.set(Math.floor(record.timestamp_sec), {
      timestampSec: record.timestamp_sec,
      count: record.count,
    });
  }

  return Array.from(latestReadingBySecond.values()).slice(-8);
}

function getTimeOnScreenRows(metrics: StreamMetrics): Array<{
  trackId: string;
  seconds: number;
}> {
  return Object.entries(metrics.time_on_screen_by_track_id)
    .map(([trackId, seconds]) => ({ trackId, seconds }))
    .sort((left, right) => right.seconds - left.seconds);
}

function formatTimestamp(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remainingSeconds}`;
}
