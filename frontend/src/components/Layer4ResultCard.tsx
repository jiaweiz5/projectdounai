"use client";

// Show Layer 4 results in the active app language. API identifiers stay stable.
import { useLocale } from "next-intl";

export type Layer4AlignmentResult = {
  status: string;
  available: boolean;
  task?: string;
  image_text_similarity?: number | null;
  threshold?: number | null;
  decision_margin?: number | null;
  label?: number | null;
  prediction?: string | null;
  risk?: string;
  model_name?: string;
  raw_logit?: number;
  calibration_case_count?: number | null;
  calibration_note?: string | null;
  text_window_count?: number;
  text_token_count?: number;
  text_tokens_covered?: number;
  text_coverage_ratio?: number;
  best_window_index?: number;
  text_scoring_method?: string;
  message?: string;
};

export type Layer4OCRResult = {
  ocr_text: string;
  ocr_confidence: number;
  ocr_item_count: number;
  message?: string;
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

// Keep every user-facing phrase here until it can be moved into your catalogs.
// These explanations describe what was checked and avoid claiming intent.
const copy = {
  zh: {
    eyebrow: "第 4 层 · 图片检查",
    title: "图片和正文是否说的是同一件事？",
    checking: "待人工复核",
    mismatch: "可能不一致",
    aligned: "基本对应",
    unavailable: "暂时无法判断",
    noImage: "还没有图片",
    partial: "部分图片未能检查",
    risk: "需要关注的程度",
    high: "较高",
    low: "较低",
    unknown: "待确认",
    conclusion: "检查结果",
    imagesChecked: "本次图片",
    image: "图片",
    score: "图文匹配分数",
    referenceLine: "判断参考线",
    margin: "与参考线的差距",
    fileSize: "文件大小",
    imageText: "图片中识别到的文字",
    imageTextEmpty: "没有识别到清晰文字",
    itemCount: "处文字",
    ocrConfidence: "文字识别把握度",
    coverage: "本次查看的正文范围",
    segments: "分段查看",
    preliminary: "这篇正文较长，系统已分段与图片比较；长文的判断标准还在验证中。现在只能作为线索，请结合图片和全文自行核对。",
    lowCoverage: "仍有一部分正文未被查看，不能据此判断整篇文字与图片是否一致。",
    scoreHelp: "匹配分数只用于比较图文内容，不是“造假概率”，也不能证明发帖者的意图。",
    marginHelp: "差距越接近 0，结果越靠近判断边界；这项判断也可能出错。",
    normalHelp: "系统只检查图片和正文的内容是否对应，不能判断图片是否经过修改，也不能证明发帖者有意误导。",
    failedHelp: "图片检查暂时未完成。其他检测结果仍可查看；请稍后重试或换一张清晰图片。",
    noImageHelp: "添加至少一张帖子图片后，才能检查图文是否对应。",
    partialHelp: "有些图片没有检查成功；已完成的结果只适用于对应的图片。",
    confirmedMismatch: "至少一张图片与正文可能不一致，请查看下方的逐张结果。",
    confirmedAligned: "已检查的图片与正文基本对应；仍建议结合原帖判断。",
  },
  en: {
    eyebrow: "LAYER 4 · IMAGE CHECK",
    title: "Does the image match the post?",
    checking: "Needs review",
    mismatch: "Possible mismatch",
    aligned: "Appears aligned",
    unavailable: "Cannot determine",
    noImage: "No image yet",
    partial: "Some images were not checked",
    risk: "Level of concern",
    high: "Higher",
    low: "Lower",
    unknown: "Unconfirmed",
    conclusion: "Result",
    imagesChecked: "Images submitted",
    image: "Image",
    score: "Image-text match score",
    referenceLine: "Decision reference",
    margin: "Distance from reference",
    fileSize: "File size",
    imageText: "Text found in the image",
    imageTextEmpty: "No readable text found",
    itemCount: "text items",
    ocrConfidence: "OCR confidence",
    coverage: "Caption reviewed",
    segments: "Text segments",
    preliminary: "This long caption was compared in segments. The long-text decision rule has not been validated yet. Treat this as a clue and check the image against the full caption yourself.",
    lowCoverage: "Part of the caption was not reviewed, so this cannot describe the whole post.",
    scoreHelp: "The match score is not a probability of deception and cannot establish intent.",
    marginHelp: "A value closer to zero is closer to the decision boundary; the decision can still be wrong.",
    normalHelp: "This checks whether the image matches the caption. It does not check whether the image was edited or establish intent.",
    failedHelp: "The image check could not finish. Other detector results remain available. Try again or use a clearer image.",
    noImageHelp: "Add at least one post image to check image-text alignment.",
    partialHelp: "Some images could not be checked. Results for completed images apply only to those images.",
    confirmedMismatch: "At least one checked image may not match the caption. Review each image below.",
    confirmedAligned: "The checked images appear to match the caption. Review the original post as well.",
  },
} as const;

// Display scores as numbers, rather than turning them into false percentages.
function formatScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "—";
}

// Translate one image's state; a provisional score is never called aligned.
function imageState(
  alignment: Layer4AlignmentResult,
  t: (typeof copy)["zh"] | (typeof copy)["en"],
): string {
  if (alignment.status === "provisional") return t.checking;
  if (alignment.status !== "completed" || alignment.label == null) {
    return t.unavailable;
  }
  return alignment.label === 1 ? t.mismatch : t.aligned;
}

// Render a short interpretation for the entire post, including partial checks.
function overallState(
  result: Layer4Result,
  t: (typeof copy)["zh"] | (typeof copy)["en"],
): string {
  if (result.status === "insufficient_data") return t.noImage;
  // Preserve a confirmed mismatch when another image remains provisional.
  if (result.risk === "high" && result.label === 1) return t.mismatch;
  if (result.status === "provisional") return t.checking;
  if (result.status === "unavailable") return t.unavailable;
  if (result.status === "partial" && result.label !== 1) return t.partial;
  if (result.label === 1) return t.mismatch;
  if (result.status === "completed" && result.label === 0) return t.aligned;
  return t.unavailable;
}

// Show the explanation most useful for this state in the selected language.
function overallExplanation(
  result: Layer4Result,
  t: (typeof copy)["zh"] | (typeof copy)["en"],
): string {
  if (result.status === "insufficient_data") return t.noImageHelp;
  if (result.status === "unavailable") return t.failedHelp;
  if (result.status === "provisional" && result.risk === "high" && result.label === 1) {
    return `${t.confirmedMismatch} ${t.preliminary}`;
  }
  if (result.status === "provisional") return t.preliminary;
  if (result.status === "partial") return t.partialHelp;
  return result.label === 1 ? t.confirmedMismatch : t.confirmedAligned;
}

export default function Layer4ResultCard({ result }: { result: Layer4Result }) {
  // Follow the same next-intl locale selected on the rest of the page.
  const isChinese = useLocale().toLowerCase().startsWith("zh");
  const t = isChinese ? copy.zh : copy.en;
  const verdict = overallState(result, t);
  const isProvisional = result.status === "provisional";
  const hasConfirmedConcern = result.risk === "high" && result.label === 1;
  const frameStyle = isProvisional || result.status === "partial"
    ? "border-amber-200 bg-amber-50 text-amber-950"
    : hasConfirmedConcern
      ? "border-rose-200 bg-rose-50 text-rose-950"
      : "border-emerald-200 bg-emerald-50 text-emerald-950";
  const badgeStyle = isProvisional || result.status === "partial"
    ? "bg-amber-100 text-amber-800"
    : hasConfirmedConcern
      ? "bg-rose-100 text-rose-800"
      : "bg-emerald-100 text-emerald-800";

  return (
    <section className={`rounded-xl border p-4 shadow-sm sm:p-6 ${frameStyle}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold tracking-widest opacity-70">{t.eyebrow}</p>
          <h2 className="mt-1 text-xl font-bold">{t.title}</h2>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeStyle}`}>
          {verdict}
        </span>
      </div>

      <dl className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg bg-white/90 p-4">
          <dt className="text-xs text-gray-500">{t.risk}</dt>
          <dd className="mt-1 font-semibold text-gray-900">
            {hasConfirmedConcern ? t.high : result.status === "completed" && result.risk === "low" ? t.low : t.unknown}
          </dd>
        </div>
        <div className="rounded-lg bg-white/90 p-4">
          <dt className="text-xs text-gray-500">{t.conclusion}</dt>
          <dd className="mt-1 font-semibold text-gray-900">{verdict}</dd>
        </div>
        <div className="rounded-lg bg-white/90 p-4">
          <dt className="text-xs text-gray-500">{t.imagesChecked}</dt>
          <dd className="mt-1 font-semibold text-gray-900">{result.image_count}</dd>
        </div>
      </dl>

      {result.images?.map((item) => {
        const alignment = item.alignment;
        const provisional = alignment.status === "provisional";
        const state = imageState(alignment, t);
        const coverage = alignment.text_coverage_ratio;
        return (
          <article key={item.index} className="mt-5 rounded-lg bg-white p-4 text-gray-900 sm:p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-semibold">{t.image} {item.index}</h3>
              <span className={`rounded-full px-3 py-1 text-xs font-semibold ${provisional ? "bg-amber-100 text-amber-800" : alignment.label === 1 ? "bg-rose-100 text-rose-800" : "bg-slate-100 text-slate-700"}`}>
                {state}
              </span>
            </div>

            <dl className="mt-4 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
              <div><dt className="text-gray-500">{t.score}</dt><dd className="font-semibold">{formatScore(alignment.image_text_similarity)}</dd></div>
              {provisional ? (
                <>
                  <div><dt className="text-gray-500">{t.coverage}</dt><dd className="font-semibold">{typeof coverage === "number" ? new Intl.NumberFormat(isChinese ? "zh-CN" : "en-US", { style: "percent", maximumFractionDigits: 1 }).format(coverage) : "—"}</dd></div>
                  <div><dt className="text-gray-500">{t.segments}</dt><dd className="font-semibold">{alignment.text_window_count ?? "—"}</dd></div>
                </>
              ) : (
                <>
                  <div><dt className="text-gray-500">{t.referenceLine}</dt><dd className="font-semibold">{formatScore(alignment.threshold)}</dd></div>
                  <div><dt className="text-gray-500">{t.margin}</dt><dd className="font-semibold">{formatScore(alignment.decision_margin)}</dd></div>
                </>
              )}
              <div><dt className="text-gray-500">{t.fileSize}</dt><dd className="font-semibold">{(item.byte_count / 1024 / 1024).toFixed(2)} MB</dd></div>
            </dl>
            <p className="mt-3 text-xs leading-relaxed text-gray-600">
              {provisional ? t.preliminary : alignment.status === "completed" ? t.marginHelp : t.failedHelp} {t.scoreHelp}
            </p>
            {provisional && typeof coverage === "number" && coverage < 1 && (
              <p className="mt-2 text-xs font-medium text-amber-800">{t.lowCoverage}</p>
            )}

            <div className="mt-4 rounded-lg bg-gray-100 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-sm font-semibold">{t.imageText}</h4>
                <span className="text-xs text-gray-500">
                  {item.ocr.ocr_item_count} {t.itemCount} · {t.ocrConfidence} {new Intl.NumberFormat(isChinese ? "zh-CN" : "en-US", { style: "percent", maximumFractionDigits: 1 }).format(item.ocr.ocr_confidence || 0)}
                </span>
              </div>
              <p className="mt-2 whitespace-pre-wrap break-words text-sm">
                {item.ocr.ocr_text || t.imageTextEmpty}
              </p>
            </div>
          </article>
        );
      })}

      <p className="mt-5 text-sm leading-relaxed opacity-80">
        {overallExplanation(result, t)} {result.status === "completed" ? t.normalHelp : ""}
      </p>
    </section>
  );
}
