"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, RefObject } from "react";
import { useTranslations } from "next-intl";

// Describe the fields returned by the independent image-reference endpoint.
type ReferenceResponse = {
  filename?: string;
  reference_filename?: string;
  layer4_reference?: Record<string, unknown>;
  message?: string;
};

// Keep the two upload slots visually consistent while their labels differ.
type ImageSlotProps = {
  title: string;
  description: string;
  chooseLabel: string;
  removeLabel: string;
  previewAlt: string;
  file: File | null;
  previewUrl: string | null;
  inputRef: RefObject<HTMLInputElement | null>;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
};

function ImageSlot({
  title,
  description,
  chooseLabel,
  removeLabel,
  previewAlt,
  file,
  previewUrl,
  inputRef,
  onChange,
  onRemove,
}: ImageSlotProps) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
      <h3 className="font-semibold text-zinc-900">{title}</h3>
      <p className="mt-1 text-sm leading-6 text-zinc-600">{description}</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={onChange}
        className="sr-only"
      />
      {/* Click this translated button to open the browser's file picker. */}
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-3 rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
      >
        {chooseLabel}
      </button>

      {file && previewUrl && (
        <div className="relative mt-3">
          {/* The browser creates this temporary URL only for the local preview. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt={previewAlt}
            className="h-52 w-full rounded-lg border border-zinc-200 bg-white object-contain"
          />
          <p className="mt-2 truncate text-xs text-zinc-600" title={file.name}>
            {file.name}
          </p>
          <button
            type="button"
            onClick={onRemove}
            aria-label={removeLabel}
            title={removeLabel}
            className="absolute right-2 top-2 flex h-9 w-9 items-center justify-center rounded-full bg-black/75 text-2xl leading-none text-white hover:bg-black focus:outline-none focus:ring-2 focus:ring-white"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

export default function Layer4ReferenceAnalyzer() {
  // Load labels and feedback in the language currently chosen in the app.
  const t = useTranslations("ReferenceTool");
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [targetPreview, setTargetPreview] = useState<string | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ReferenceResponse | null>(null);
  const targetInputRef = useRef<HTMLInputElement>(null);
  const referenceInputRef = useRef<HTMLInputElement>(null);

  // Release temporary browser image URLs when a preview changes or the page closes.
  useEffect(() => {
    const targetUrl = targetPreview;
    const referenceUrl = referencePreview;
    return () => {
      if (targetUrl) URL.revokeObjectURL(targetUrl);
      if (referenceUrl) URL.revokeObjectURL(referenceUrl);
    };
  }, [targetPreview, referencePreview]);

  // Validate the selected image and update the matching preview slot.
  function handleImageChange(
    kind: "target" | "reference",
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0] ?? null;
    // Reset the input so selecting the same file again will still be detected.
    event.target.value = "";
    setError("");
    setResult(null);
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError(t("invalidImage"));
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setError(t("fileTooLarge"));
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    if (kind === "target") {
      setTargetFile(file);
      setTargetPreview(previewUrl);
    } else {
      setReferenceFile(file);
      setReferencePreview(previewUrl);
    }
  }

  // Remove one selected image so it will not be sent to the comparison endpoint.
  function removeImage(kind: "target" | "reference") {
    if (kind === "target") {
      setTargetFile(null);
      setTargetPreview(null);
      if (targetInputRef.current) targetInputRef.current.value = "";
    } else {
      setReferenceFile(null);
      setReferencePreview(null);
      if (referenceInputRef.current) referenceInputRef.current.value = "";
    }
    setResult(null);
    setError("");
  }

  // Send both selected images to the existing reference-comparison API route.
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isLoading) return;
    if (!targetFile || !referenceFile) {
      setError(t("selectBothImages"));
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    try {
      // The backend expects these exact multipart field names.
      const formData = new FormData();
      formData.append("file", targetFile);
      formData.append("reference_file", referenceFile);

      const response = await fetch("/api/analyze-layer4-reference", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json().catch(() => ({}))) as
        | ReferenceResponse
        | { detail?: string; error?: string };

      if (!response.ok) {
        if (response.status === 413) throw new Error(t("fileTooLarge"));
        if (response.status === 415) throw new Error(t("invalidImage"));
        throw new Error(t("comparisonFailed"));
      }
      const parsed = payload as ReferenceResponse;
      if (
        !parsed.layer4_reference ||
        typeof parsed.layer4_reference !== "object"
      ) {
        throw new Error(t("invalidResponse"));
      }
      setResult(parsed);
    } catch (requestError) {
      // Keep translated validation messages; use a translated fallback otherwise.
      const knownErrors = [t("fileTooLarge"), t("invalidImage"), t("comparisonFailed"), t("invalidResponse")];
      setError(
        requestError instanceof Error && knownErrors.includes(requestError.message)
          ? requestError.message
          : t("comparisonFailed"),
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <p className="mb-4 text-sm leading-6 text-zinc-600">{t("workflowHint")}</p>
      <form onSubmit={handleSubmit} className="grid gap-4">
        <div className="grid gap-4 md:grid-cols-2">
          <ImageSlot
            title={t("targetTitle")}
            description={t("targetDescription")}
            chooseLabel={t("chooseImage")}
            removeLabel={t("removeTarget")}
            previewAlt={t("targetPreviewAlt")}
            file={targetFile}
            previewUrl={targetPreview}
            inputRef={targetInputRef}
            onChange={(event) => handleImageChange("target", event)}
            onRemove={() => removeImage("target")}
          />
          <ImageSlot
            title={t("referenceTitle")}
            description={t("referenceDescription")}
            chooseLabel={t("chooseImage")}
            removeLabel={t("removeReference")}
            previewAlt={t("referencePreviewAlt")}
            file={referenceFile}
            previewUrl={referencePreview}
            inputRef={referenceInputRef}
            onChange={(event) => handleImageChange("reference", event)}
            onRemove={() => removeImage("reference")}
          />
        </div>

        <button
          type="submit"
          disabled={!targetFile || !referenceFile || isLoading}
          className="w-full rounded-lg bg-zinc-900 px-5 py-3 font-semibold text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-400"
        >
          {isLoading ? t("comparing") : t("compareButton")}
        </button>
      </form>

      {error && (
        <p role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">
          {error}
        </p>
      )}

      {result?.layer4_reference && (
        <section className="mt-4 rounded-lg bg-zinc-50 p-4">
          <h3 className="font-semibold text-zinc-900">{t("resultTitle")}</h3>
          <p className="mt-1 text-sm text-zinc-600">{t("resultHint")}</p>
          <pre className="mt-3 overflow-x-auto rounded-lg bg-zinc-950 p-4 text-sm text-green-300">
            {JSON.stringify(result.layer4_reference, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
