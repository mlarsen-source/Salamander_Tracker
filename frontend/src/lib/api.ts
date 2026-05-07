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

export async function processVideo(file: File): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append("video", file);

  const response = await fetch(`${API_BASE}/api/process-video`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to process video");
  }

  return response.json();
}
