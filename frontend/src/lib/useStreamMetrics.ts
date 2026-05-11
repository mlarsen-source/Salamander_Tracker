/**
 * React hook that polls the backend for live stream metrics.
 *
 * Starts polling shortly after `streamUrl` becomes available so the backend
 * has time to initialize the streamer; stops automatically when the backend
 * reports the stream has ended.
 */
import { useEffect, useRef, useState } from "react";

import { getStreamMetrics, type StreamMetrics } from "@/lib/api";

const POLL_INTERVAL_MS = 500;
const POLL_START_DELAY_MS = 1000;

type UseStreamMetricsResult = {
  metrics: StreamMetrics | null;
  /** True until the backend reports `is_streaming === false`. */
  isStreaming: boolean;
  reset: () => void;
};

export function useStreamMetrics(
  streamUrl: string | null
): UseStreamMetricsResult {
  const [metrics, setMetrics] = useState<StreamMetrics | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const intervalHandleRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!streamUrl) {
      return;
    }

    setIsStreaming(true);

    const fetchAndStoreMetrics = async () => {
      try {
        const latest = await getStreamMetrics();
        setMetrics(latest);
        if (!latest.is_streaming) {
          setIsStreaming(false);
        }
      } catch (error) {
        // The first few polls can race the backend startup; ignore that
        // specific message but surface any other failure to the console.
        const message = error instanceof Error ? error.message : String(error);
        if (!message.includes("No stream in progress")) {
          console.error("Failed to fetch metrics:", error);
        }
      }
    };

    const startupDelayHandle = setTimeout(() => {
      void fetchAndStoreMetrics();
      intervalHandleRef.current = setInterval(
        fetchAndStoreMetrics,
        POLL_INTERVAL_MS
      );
    }, POLL_START_DELAY_MS);

    return () => {
      clearTimeout(startupDelayHandle);
      if (intervalHandleRef.current) {
        clearInterval(intervalHandleRef.current);
        intervalHandleRef.current = null;
      }
    };
  }, [streamUrl]);

  const reset = () => {
    setMetrics(null);
    setIsStreaming(false);
  };

  return { metrics, isStreaming, reset };
}
