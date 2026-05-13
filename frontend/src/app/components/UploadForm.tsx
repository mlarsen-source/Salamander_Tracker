"use client";

/**
 * Upload + control form: pick a video, start a stream, stop a running one.
 *
 * State for the file selection lives here; everything else (whether the
 * stream is starting/running, the current error message, the stop handler)
 * is owned by the parent and passed in.
 */
import { useState, type FormEvent } from "react";

import styles from "./UploadForm.module.css";

type UploadFormProps = {
  isStarting: boolean;
  isStreaming: boolean;
  errorMessage: string | null;
  onStart: (videoFile: File) => void | Promise<void>;
  onStop: () => void | Promise<void>;
};

export function UploadForm({
  isStarting,
  isStreaming,
  errorMessage,
  onStart,
  onStop,
}: UploadFormProps) {
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);

  const isBusy = isStarting || isStreaming;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (selectedVideoFile) {
      void onStart(selectedVideoFile);
    }
  };

  const submitButtonLabel = isStarting
    ? "Starting..."
    : isStreaming
      ? "Streaming..."
      : "Start Stream";

  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={styles.uploadForm}>
        <label className={styles.filePicker}>
          <span className={styles.filePickerLabel}>Source Footage</span>
          <input
            type="file"
            accept="video/*"
            disabled={isBusy}
            onChange={(event) => {
              setSelectedVideoFile(event.target.files?.[0] ?? null);
            }}
          />
        </label>
        <div className={styles.fileName} title={selectedVideoFile?.name}>
          {selectedVideoFile?.name ?? "No video selected"}
        </div>
        <div className={styles.actions}>
          <button
            type="submit"
            disabled={isBusy || !selectedVideoFile}>
            {submitButtonLabel}
          </button>
          {isStreaming && (
          <button
            type="button"
            className={styles.stopButton}
            onClick={() => void onStop()}>
            Stop
          </button>
          )}
        </div>
      </form>
      {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}
    </>
  );
}
