/**
 * Typed client for the Salamander Tracker Flask backend.
 *
 * One function per backend endpoint. Network and JSON-parsing concerns live
 * here so React components can stay focused on rendering.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

/** Batch processing of a full video can take many minutes on CPU. */
const PROCESS_VIDEO_TIMEOUT_MS = 15 * 60 * 1000;

// ---------------------------------------------------------------------------
// Types returned by the backend
// ---------------------------------------------------------------------------

export type FrameDetectionRecord = {
  frame: number;
  timestamp_sec: number;
  count: number;
};

export type StreamStartResponse = {
  run_id: string;
  stream_url: string;
  metrics_url: string;
};

export type StreamMetrics = {
  frame_count: number;
  fps: number;
  is_streaming: boolean;
  detection_count_over_time: FrameDetectionRecord[];
  time_on_screen_by_track_id: Record<string, number>;
};

export type ProcessVideoResponse = {
  run_id: string;
  annotated_video_url: string;
  metrics: {
    detection_count_over_time: FrameDetectionRecord[];
    time_on_screen_by_track_id: Record<string, number>;
  };
};

// Backwards-compat alias kept so imports of `StreamResponse` keep working.
export type StreamResponse = StreamStartResponse;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Wrap fetch with a friendly error message extracted from the response body. */
async function fetchOrThrow(
  url: string,
  init: RequestInit,
  fallbackError: string,
): Promise<Response> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || fallbackError);
  }
  return response;
}

function buildVideoFormData(file: File): FormData {
  const formData = new FormData();
  formData.append("video", file);
  return formData;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Tell the backend to open the uploaded video and prepare a live MJPEG stream. */
export async function startStream(file: File): Promise<StreamStartResponse> {
  const response = await fetchOrThrow(
    `${API_BASE_URL}/api/stream-video`,
    { method: "POST", body: buildVideoFormData(file) },
    "Failed to start stream",
  );
  return response.json();
}

/** Fetch a snapshot of the live stream's metrics for the dashboard. */
export async function getStreamMetrics(): Promise<StreamMetrics> {
  const response = await fetchOrThrow(
    `${API_BASE_URL}/api/stream-metrics`,
    {},
    "Failed to get metrics",
  );
  return response.json();
}

/** Ask the backend to halt the in-flight stream. Safe to call when nothing is streaming. */
export async function stopStream(): Promise<void> {
  await fetchOrThrow(
    `${API_BASE_URL}/api/stream-stop`,
    { method: "POST" },
    "Failed to stop stream",
  );
}

/**
 * Batch-process a full video and get back a URL to the annotated MP4.
 * Currently unused by the UI but kept available for direct API testing.
 */
export async function processVideo(file: File): Promise<ProcessVideoResponse> {
  const abortController = new AbortController();
  const timeoutHandle = setTimeout(
    () => abortController.abort(),
    PROCESS_VIDEO_TIMEOUT_MS,
  );

  try {
    const response = await fetchOrThrow(
      `${API_BASE_URL}/api/process-video`,
      {
        method: "POST",
        body: buildVideoFormData(file),
        signal: abortController.signal,
      },
      "Failed to process video",
    );
    return response.json();
  } finally {
    clearTimeout(timeoutHandle);
  }
}
