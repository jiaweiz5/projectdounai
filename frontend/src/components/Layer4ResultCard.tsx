"use client";

// These types describe the Layer 4 section returned by the FastAPI backend.
// Keeping them here lets page.tsx reuse the exact same response definition.
export type Layer4AlignmentResult = {
  status: string;
  available: boolean;
  task: string;
  image_text_similarity: number;
  threshold: number;
  decision_margin: number;
  label: number;
  prediction: string;
  risk: string;
  model_name?: string;
  raw_logit?: number;
  calibration_case_count?: number;
  calibration_note?: string;
};

export type Layer4OCRResult = {
  ocr_text: string;
  ocr_confidence: number;
  ocr_item_count: number;
};

export type Layer4ImageResult = {
  index: number;
  byte_count: number;
  alignment: Layer4AlignmentResult;
  ocr: Layer4OCRResult;
};

export type Layer4Result = {
  status: string;
  available: boolean;
  label: number | null;
  prediction: string | null;
  risk: string;
  image_count: number;
  message?: string;
  images?: Layer4ImageResult[];
};

type Layer4ResultCardProps = {
  result: Layer4Result;
};

function formatScore(value: number | null | undefined) {
  if (typeof value !== "number") return "Not available";
  return value.toFixed(4);
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} bytes`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

export default function Layer4ResultCard({
  result,
}: Layer4ResultCardProps) {
  // No uploaded image is a valid request, but Layer 4 cannot make a decision.
  if (result.status === "insufficient_data") {
    return (
      <section className="rounded-xl border border-amber-200 bg-amber-50 p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-amber-950">
          Layer 4: Image Verification
        </h2>

        <p className="mt-2 text-sm text-amber-800">
          {result.message ?? "Upload a post image to run Layer 4."}
        </p>
      </section>
    );
  }

  const suspicious = result.label === 1;
  const panelClasses = suspicious
    ? "border-red-200 bg-red-50"
    : "border-emerald-200 bg-emerald-50";
  const headingClasses = suspicious ? "text-red-950" : "text-emerald-950";
  const badgeClasses = suspicious
    ? "bg-red-100 text-red-800"
    : "bg-emerald-100 text-emerald-800";
  const resultLabel = suspicious ? "Image-text mismatch" : "Aligned";

  return (
    <section className={`rounded-xl border p-6 shadow-sm ${panelClasses}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Layer 4
          </p>

          <h2 className={`mt-1 text-xl font-semibold ${headingClasses}`}>
            Image–Text Alignment
          </h2>
        </div>

        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeClasses}`}>
          {resultLabel}
        </span>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg bg-white/80 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">Risk</p>
          <p className="mt-1 font-semibold capitalize text-zinc-900">
            {result.risk}
          </p>
        </div>

        <div className="rounded-lg bg-white/80 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">
            Prediction
          </p>
          <p className="mt-1 font-semibold text-zinc-900">{resultLabel}</p>
        </div>

        <div className="rounded-lg bg-white/80 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">
            Images checked
          </p>
          <p className="mt-1 font-semibold text-zinc-900">
            {result.image_count}
          </p>
        </div>
      </div>

      <div className="mt-5 space-y-4">
        {(result.images ?? []).map((image) => {
          const alignment = image.alignment;
          const imageMismatch = alignment.label === 1;

          return (
            <article
              key={image.index}
              className="rounded-lg border border-white/80 bg-white p-5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-semibold text-zinc-900">
                  Image {image.index}
                </h3>

                <span
                  className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                    imageMismatch
                      ? "bg-red-100 text-red-700"
                      : "bg-emerald-100 text-emerald-700"
                  }`}
                >
                  {imageMismatch ? "Mismatch" : "Aligned"}
                </span>
              </div>

              <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <dt className="text-zinc-500">Similarity</dt>
                  <dd className="font-semibold text-zinc-900">
                    {formatScore(alignment.image_text_similarity)}
                  </dd>
                </div>

                <div>
                  <dt className="text-zinc-500">Threshold</dt>
                  <dd className="font-semibold text-zinc-900">
                    {formatScore(alignment.threshold)}
                  </dd>
                </div>

                <div>
                  <dt className="text-zinc-500">Decision margin</dt>
                  <dd className="font-semibold text-zinc-900">
                    {alignment.decision_margin >= 0 ? "+" : ""}
                    {formatScore(alignment.decision_margin)}
                  </dd>
                </div>

                <div>
                  <dt className="text-zinc-500">File size</dt>
                  <dd className="font-semibold text-zinc-900">
                    {formatBytes(image.byte_count)}
                  </dd>
                </div>
              </dl>

              <p className="mt-3 text-xs text-zinc-500">
                Positive margins favor alignment; negative margins favor a
                mismatch. Smaller absolute margins are closer to the calibrated
                decision boundary.
              </p>

              <div className="mt-4 rounded-lg bg-zinc-100 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h4 className="text-sm font-semibold text-zinc-900">
                    Text detected in image
                  </h4>

                  <span className="text-xs text-zinc-500">
                    {image.ocr.ocr_item_count} item
                    {image.ocr.ocr_item_count === 1 ? "" : "s"} · OCR confidence{" "}
                    {formatScore(image.ocr.ocr_confidence)}
                  </span>
                </div>

                <p className="mt-2 whitespace-pre-wrap text-sm text-zinc-700">
                  {image.ocr.ocr_text || "No readable text was detected."}
                </p>
              </div>
            </article>
          );
        })}
      </div>

      {result.message && (
        <p className="mt-4 text-xs text-zinc-600">{result.message}</p>
      )}
    </section>
  );
}
