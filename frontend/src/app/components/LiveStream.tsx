"use client";

/**
 * Renders the live MJPEG feed served from the backend.
 *
 * The browser handles the multipart stream natively when it's pointed at
 * an `<img>` tag, so all this component has to do is supply the right URL.
 */
import styles from "./LiveStream.module.css";

type LiveStreamProps = {
  streamUrl: string;
};

export function LiveStream({ streamUrl }: LiveStreamProps) {
  return (
    <div className={styles.streamFrame}>
      <div className={styles.scanLine} aria-hidden="true" />
      <img
        key={streamUrl}
        src={streamUrl}
        alt="Live salamander detection stream"
        className={styles.streamImage}
      />
    </div>
  );
}
