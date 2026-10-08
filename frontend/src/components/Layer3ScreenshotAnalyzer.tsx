"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { useTranslations } from "next-intl";

// Shape of one comment recognized by the backend OCR service.
type OCRComment = {
  text: string;
  confidence: number;
  box?: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
};

// The screenshot endpoint returns both OCR and a preliminary Layer 3 result.
// This component uses the OCR comments; the main Analyze request runs the
// combined Layers 1–4 analysis and produces the final report.
type ScreenshotResponse = {
  filename?: string;
  ocr: {
    line_count: number;
    comment_count: number;
    comments: OCRComment[];
  };
  message?: string;
};

type Props = {
  // Pass recognized comment text back to the page's shared comments state.
  onCommentsExtracted: (comments: string[]) => void;
};

export default function Layer3ScreenshotAnalyzer({
  onCommentsExtracted,
}: Props) {
  // Load the screenshot workflow copy in the language selected by the user.
  const t = useTranslations("Screenshot");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ScreenshotResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // The visible button opens this hidden file input's native file picker.
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Keep the temporary URL so it can be released when a new file is selected.
  const previewUrlRef = useRef<string | null>(null);

  // Release browser memory used by the local image preview on change/unmount.
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  // Validate the chosen file and create a temporary URL for its preview.
  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    // Clear the input so the same screenshot can be selected again if needed.
    event.target.value = "";
    setResult(null);
    setError("");

    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }
    if (!file.type.startsWith("image/")) {
      setSelectedFile(null);
      setError(t("invalidImage"));
      event.target.value = "";
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setSelectedFile(null);
      setError(t("fileTooLarge"));
      event.target.value = "";
      return;
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    previewUrlRef.current = nextPreviewUrl;
    setSelectedFile(file);
    setPreviewUrl(nextPreviewUrl);
  }

  // Clear the selected screenshot and its preview without sending it for OCR.
  function removeSelectedFile() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
    setSelectedFile(null);
    setResult(null);
    setError("");
  }

  // Upload the screenshot, then give extracted comment text to the main page.
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile || isLoading) return;

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      // FormData sends the image as multipart data; the browser sets its boundary.
      const formData = new FormData();
      formData.append("file", selectedFile);

      // Use the same-origin Next.js proxy so the backend token stays server-side.
      const response = await fetch("/api/analyze-screenshot", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json().catch(() => ({}))) as
        | ScreenshotResponse
        | { detail?: string; error?: string };

      if (!response.ok) {
        // Convert known backend statuses to messages in the selected language.
        if (response.status === 413) throw new Error(t("fileTooLarge"));
        if (response.status === 415) throw new Error(t("invalidImage"));
        if (response.status === 422) throw new Error(t("ocrFailed"));
        throw new Error(t("processingFailed"));
      }

      if (!("ocr" in payload) || !Array.isArray(payload.ocr?.comments)) {
        throw new Error(t("invalidResponse"));
      }

      const extracted = payload.ocr.comments
        .map((comment) => comment.text)
        .filter((comment): comment is string => typeof comment === "string");
      onCommentsExtracted(extracted);
      setResult(payload);
    } catch (requestError) {
      // Preserve deliberate translated errors; translate network failures too.
      const knownTranslatedErrors = [
        t("fileTooLarge"),
        t("invalidImage"),
        t("ocrFailed"),
        t("invalidResponse"),
      ];
      const message =
        requestError instanceof Error &&
        knownTranslatedErrors.includes(requestError.message)
          ? requestError.message
          : t("processingFailed");
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
      <h3 className="font-semibold text-zinc-900">{t("title")}</h3>
      <p className="mt-1 text-sm leading-6 text-zinc-600">
        {t("description")}
      </p>

      <form onSubmit={handleSubmit} className="mt-4 grid gap-3">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          aria-label={t("chooseScreenshot")}
          className="sr-only"
        />
        {/* Keep the native picker behavior while showing a translated button. */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-fit rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-600"
        >
          {t("chooseScreenshot")}
        </button>
        {previewUrl && (
          <div className="relative">
            {/* A regular img is appropriate for this temporary local preview URL. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt={t("screenshotPreviewAlt")}
              className="max-h-80 w-full rounded-lg bg-zinc-900 object-contain"
            />
            <button
              type="button"
              onClick={removeSelectedFile}
              aria-label={t("removeScreenshot")}
              className="absolute right-2 top-2 flex h-9 w-9 items-center justify-center rounded-full bg-black/75 text-2xl leading-none text-white hover:bg-black focus:outline-none focus:ring-2 focus:ring-white"
            >
              ×
            </button>
          </div>
        )}
        <button
          type="submit"
          disabled={!selectedFile || isLoading}
          className="w-fit rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? t("extracting") : t("extractButton")}
        </button>
      </form>

      {error && (
        <p role="alert" className="mt-3 rounded bg-red-50 p-3 text-sm text-red-700">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-4 rounded-lg bg-white p-3 text-sm text-zinc-700">
          {result.ocr.comments.length === 0 ? (
            <p className="font-medium">
              {t("noneExtracted")}
            </p>
          ) : (
            <p className="font-medium">
              {t("extractedSummary", { count: result.ocr.comments.length })}
            </p>
          )}
          {result.ocr.comments.length === 0 ? (
            <p className="mt-2 text-amber-800">
              {t("noCommentsHint")}
            </p>
          ) : (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {result.ocr.comments.map((comment, index) => (
                <li key={`${comment.text}-${index}`}>
                  {comment.text}
                  <span className="ml-2 text-xs text-zinc-500">
                    {t("ocrConfidence", {
                      percent: (comment.confidence * 100).toFixed(0),
                    })}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
