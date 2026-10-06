// This component previews a local blob URL, so Next image optimization
// is not useful for this temporary browser-only image.
/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, CSSProperties, FormEvent } from "react";


// The API response types make mistakes easier to catch while editing the UI.
type OCRComment = {
  text: string;
  confidence: number;
  box: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
};

type Layer3Result = {
  status: string;
  available: boolean;
  score: number | null;
  threshold: number | null;
  label: number | null;
  prediction?: "normal" | "coordinated";
  risk: "low" | "medium" | "high" | "unknown";
  comment_count: number;
  message?: string;
};

type ScreenshotResponse = {
  filename: string;
  ocr: {
    line_count: number;
    comment_count: number;
    comments: OCRComment[];
  };
  layer3: Layer3Result;
  message: string;
};


// Local development defaults to port 8001. In deployment, define
// NEXT_PUBLIC_API_URL in the frontend environment settings.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8001";


export default function Layer3ScreenshotAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  // Remember the current browser preview URL without causing another render.
  // We use this reference to release the URL when it is no longer needed.
  const previewUrlRef = useRef<string | null>(null);

  const [result, setResult] = useState<ScreenshotResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

    // This effect only performs cleanup when the component leaves the page.
  // State changes happen in the file-selection handler instead of the effect.
  useEffect(() => {
    return () => {
      const currentPreviewUrl = previewUrlRef.current;

      if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
      }
    };
  }, []);

    function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    // Get the first selected file, or null if the selection was cleared.
    const file = event.target.files?.[0] ?? null;

    // Reset the previous analysis whenever the screenshot changes.
    setResult(null);
    setError(null);

    // Release the old temporary browser URL before creating another one.
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }

    // Remove the previous preview while validating the new selection.
    setPreviewUrl(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    // Reject non-image files before sending anything to the backend.
    if (!file.type.startsWith("image/")) {
      setSelectedFile(null);
      setError("Please choose a PNG, JPEG, WEBP, or another image file.");
      return;
    }

    // Create a temporary local URL so the browser can preview the image.
    const nextPreviewUrl = URL.createObjectURL(file);

    // Remember both the actual file and its temporary preview URL.
    previewUrlRef.current = nextPreviewUrl;
    setSelectedFile(file);
    setPreviewUrl(nextPreviewUrl);
  }
    

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Choose a comment-section screenshot first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // File uploads use multipart/form-data. The browser automatically adds
      // the correct boundary, so do not manually set Content-Type here.
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_BASE_URL}/analyze-screenshot`,
        {
          method: "POST",
          body: formData,
        },
      );

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "The screenshot could not be analyzed.");
      }

      setResult(payload as ScreenshotResponse);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The screenshot could not be analyzed.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  const scorePercent =
    result?.layer3.score == null
      ? null
      : (result.layer3.score * 100).toFixed(1);

  return (
    <section style={styles.card}>
      <h2 style={styles.heading}>Comment Coordination Check</h2>
      <p style={styles.description}>
        Upload a screenshot cropped to the Xiaohongshu comment section. At
        least five visible comments are required.
      </p>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          aria-label="Choose comment screenshot"
        />

        {previewUrl && (
          <img
            src={previewUrl}
            alt="Selected comment screenshot preview"
            style={styles.preview}
          />
        )}

        <button
          type="submit"
          disabled={!selectedFile || isLoading}
          style={{
            ...styles.button,
            opacity: !selectedFile || isLoading ? 0.55 : 1,
          }}
        >
          {isLoading ? "Analyzing screenshot…" : "Analyze screenshot"}
        </button>
      </form>

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.resultPanel}>
          <h3 style={styles.resultHeading}>Layer 3 result</h3>

          <dl style={styles.metrics}>
            <div>
              <dt>Prediction</dt>
              <dd>{result.layer3.prediction ?? result.layer3.status}</dd>
            </div>
            <div>
              <dt>Risk</dt>
              <dd>{result.layer3.risk}</dd>
            </div>
            <div>
              <dt>Coordination score</dt>
              <dd>{scorePercent == null ? "Unavailable" : `${scorePercent}%`}</dd>
            </div>
            <div>
              <dt>Comments analyzed</dt>
              <dd>{result.layer3.comment_count}</dd>
            </div>
          </dl>

          {result.layer3.message && (
            <p style={styles.notice}>{result.layer3.message}</p>
          )}

          <h3 style={styles.resultHeading}>Extracted comments</h3>

          <ol style={styles.commentList}>
            {result.ocr.comments.map((comment, index) => (
              <li key={`${comment.text}-${index}`} style={styles.commentItem}>
                <span>{comment.text}</span>
                <small style={styles.confidence}>
                  OCR {(comment.confidence * 100).toFixed(1)}%
                </small>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}


// Inline styles keep this example self-contained. You can move these values
// into your existing Tailwind or CSS system after the component is working.
const styles: Record<string, CSSProperties> = {
  card: {
    width: "100%",
    maxWidth: 720,
    margin: "32px auto",
    padding: 24,
    border: "1px solid #e5e7eb",
    borderRadius: 16,
    background: "#ffffff",
    color: "#111827",
  },
  heading: {
    margin: 0,
    fontSize: 26,
  },
  description: {
    color: "#4b5563",
    lineHeight: 1.6,
  },
  form: {
    display: "grid",
    gap: 16,
  },
  preview: {
    width: "100%",
    maxHeight: 480,
    objectFit: "contain",
    borderRadius: 12,
    background: "#111827",
  },
  button: {
    border: 0,
    borderRadius: 10,
    padding: "12px 18px",
    background: "#2563eb",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
  },
  error: {
    marginTop: 16,
    padding: 12,
    borderRadius: 10,
    background: "#fef2f2",
    color: "#b91c1c",
  },
  resultPanel: {
    marginTop: 24,
    paddingTop: 20,
    borderTop: "1px solid #e5e7eb",
  },
  resultHeading: {
    marginTop: 16,
    marginBottom: 12,
  },
  metrics: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
    gap: 12,
  },
  notice: {
    padding: 12,
    borderRadius: 10,
    background: "#fffbeb",
    color: "#92400e",
  },
  commentList: {
    display: "grid",
    gap: 10,
    paddingLeft: 24,
  },
  commentItem: {
    padding: 12,
    borderRadius: 10,
    background: "#f9fafb",
  },
  confidence: {
    display: "block",
    marginTop: 4,
    color: "#6b7280",
  },
};
