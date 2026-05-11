export type StreamResponse = {
  run_id: string;
  stream_url: string;
  metrics_url: string;
};

export type StreamMetrics = {
  frame_count: number;
  fps: number;
  is_streaming: boolean;
  detection_count_over_time: Array<{
    frame: number;
    timestamp_sec: number;
    count: number;
  }>;
  time_on_screen_by_track_id: Record<string, number>;
};

export type ProcessResponse = {
  run_id: string;
  annotated_video_url: string;
  metrics: {
    detection_count_over_time: Array<{
      frame: number;
      timestamp_sec: number;
      count: number;
    }>;
    time_on_screen_by_track_id: Record<string, number>;
  };
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

// 15 minute timeout for video processing
const PROCESS_VIDEO_TIMEOUT_MS = 15 * 60 * 1000;

export async function startStream(file: File): Promise<StreamResponse> {
  const formData = new FormData();
  formData.append("video", file);

  const response = await fetch(`${API_BASE}/api/stream-video`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to start stream");
  }

  return response.json();
}

export async function getStreamMetrics(): Promise<StreamMetrics> {
  const response = await fetch(`${API_BASE}/api/stream-metrics`);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to get metrics");
  }

  return response.json();
}

export async function stopStream(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/stream-stop`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error("Failed to stop stream");
  }
}

export async function processVideo(file: File): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append("video", file);

  const controller = new AbortController();
  const timeoutId = setTimeout(
    () => controller.abort(),
    PROCESS_VIDEO_TIMEOUT_MS
  );

  try {
    const response = await fetch(`${API_BASE}/api/process-video`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Failed to process video");
    }

    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}
