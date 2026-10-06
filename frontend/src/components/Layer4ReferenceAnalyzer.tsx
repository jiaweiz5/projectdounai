"use client";

import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

type ReferenceFeatures = {
  structural_difference: number;
  mean_gray_difference: number;
  mean_color_difference: number;
  p99_5_color_difference: number;
  changed_pixel_ratio: number;
  largest_change_ratio: number;
};

type ReferenceResult = {
  status: string;
  available: boolean;
  task: string;
  edit_probability: number;
  threshold: number;
  label: number;
  prediction: string;
  risk: string;
  reference_similarity: number;
  changed_region_count: number;
  target_size: [number, number];
  reference_size: [number, number];
  features: ReferenceFeatures;
  training_note?: string;
};

type ReferenceResponse = {
  filename: string;
  reference_filename: string;
  layer4_reference: ReferenceResult;
  message?: string;
};

function formatScore(value: number) {
  return value.toFixed(4);
}

function validateImage(file: File | undefined) {
  if (!file) return "Select an image file.";

  if (!file.type.startsWith("image/")) {
    return "The selected file must be an image.";
  }

  if (file.size > MAX_IMAGE_BYTES) {
    return "Each image must be 12 MB or smaller.";
  }

  return null;
}

export default function Layer4ReferenceAnalyzer() {
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [result, setResult] = useState<ReferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleTargetFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    const validationError = validateImage(file);

    if (validationError) {
      setTargetFile(null);
      setError(validationError);
      event.target.value = "";
      return;
    }

    setTargetFile(file ?? null);
    setResult(null);
    setError("");
  }

  function handleReferenceFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    const validationError = validateImage(file);

    if (validationError) {
      setReferenceFile(null);
      setError(validationError);
      event.target.value = "";
      return;
    }

    setReferenceFile(file ?? null);
    setResult(null);
    setError("");
  }

  async function compareImages(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (loading) return;

    if (!targetFile || !referenceFile) {
      setError("Select both the suspected image and its trusted reference.");
      return;
    }

    // FormData preserves the two files as multipart uploads for FastAPI.
    const formData = new FormData();
    formData.append("file", targetFile);
    formData.append("reference_file", referenceFile);

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/analyze-layer4-reference", {
        method: "POST",
        body: formData,
      });

      const data = (await response.json()) as ReferenceResponse & {
        error?: string;
      };

      if (!response.ok) {
        throw new Error(data.error ?? "Reference comparison unavailable.");
      }

      setResult(data);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Reference comparison unavailable.",
      );
    } finally {
      setLoading(false);
    }
  }

  const analysis = result?.layer4_reference;
  const possibleEditing = analysis?.label === 1;

  return (
    <section className="rounded-xl bg-white p-6 shadow-sm">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Layer 4
        </p>

        <h2 className="mt-1 text-xl font-semibold text-zinc-900">
          Reference Image Comparison
        </h2>

        <p className="mt-2 text-sm text-zinc-600">
          Compare a suspected image with a trusted original version of the same
          image. This tool cannot work reliably without a genuine reference.
        </p>
      </div>

      <form onSubmit={compareImages} className="mt-6 space-y-5">
        <div className="grid gap-4 md:grid-cols-2">
          <label className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
            <span className="block font-semibold text-zinc-900">
              Suspected or edited image
            </span>

            <span className="mt-1 block text-xs text-zinc-500">
              This is the image you want to check.
            </span>

            <input
              type="file"
              accept="image/*"
              onChange={handleTargetFile}
              className="mt-4 block w-full text-sm text-zinc-700"
            />

            {targetFile && (
              <span className="mt-2 block text-xs font-medium text-zinc-700">
                Selected: {targetFile.name}
              </span>
            )}
          </label>

          <label className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
            <span className="block font-semibold text-zinc-900">
              Trusted reference image
            </span>

            <span className="mt-1 block text-xs text-zinc-500">
              Use a known original of the same image.
            </span>

            <input
              type="file"
              accept="image/*"
              onChange={handleReferenceFile}
              className="mt-4 block w-full text-sm text-zinc-700"
            />

            {referenceFile && (
              <span className="mt-2 block text-xs font-medium text-zinc-700">
                Selected: {referenceFile.name}
              </span>
            )}
          </label>
        </div>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !targetFile || !referenceFile}
          className="w-full rounded-lg bg-zinc-900 px-5 py-3 font-semibold text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Comparing images..." : "Compare with reference"}
        </button>
      </form>

      {analysis && (
        <div
          className={`mt-6 rounded-xl border p-5 ${
            possibleEditing
              ? "border-red-200 bg-red-50"
              : "border-emerald-200 bg-emerald-50"
          }`}
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-sm text-zinc-600">Reference result</p>
              <h3 className="mt-1 text-lg font-semibold text-zinc-900">
                {possibleEditing ? "Possible editing detected" : "No edit detected"}
              </h3>
            </div>

            <span
              className={`rounded-full px-3 py-1 text-sm font-semibold ${
                possibleEditing
                  ? "bg-red-100 text-red-800"
                  : "bg-emerald-100 text-emerald-800"
              }`}
            >
              {analysis.risk} risk
            </span>
          </div>

          <dl className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg bg-white/80 p-3">
              <dt className="text-xs text-zinc-500">Editing probability</dt>
              <dd className="mt-1 font-semibold text-zinc-900">
                {formatScore(analysis.edit_probability)}
              </dd>
            </div>

            <div className="rounded-lg bg-white/80 p-3">
              <dt className="text-xs text-zinc-500">Decision threshold</dt>
              <dd className="mt-1 font-semibold text-zinc-900">
                {formatScore(analysis.threshold)}
              </dd>
            </div>

            <div className="rounded-lg bg-white/80 p-3">
              <dt className="text-xs text-zinc-500">Reference similarity</dt>
              <dd className="mt-1 font-semibold text-zinc-900">
                {formatScore(analysis.reference_similarity)}
              </dd>
            </div>

            <div className="rounded-lg bg-white/80 p-3">
              <dt className="text-xs text-zinc-500">Changed regions</dt>
              <dd className="mt-1 font-semibold text-zinc-900">
                {analysis.changed_region_count}
              </dd>
            </div>
          </dl>

          <div className="mt-4 rounded-lg bg-white/80 p-4 text-sm text-zinc-700">
            <p>
              Target size: {analysis.target_size[0]} × {analysis.target_size[1]}
            </p>
            <p className="mt-1">
              Reference size: {analysis.reference_size[0]} ×{" "}
              {analysis.reference_size[1]}
            </p>
          </div>

          {result?.message && (
            <p className="mt-4 text-xs text-zinc-600">{result.message}</p>
          )}

          {analysis.training_note && (
            <p className="mt-2 text-xs text-zinc-500">
              {analysis.training_note}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
