"use client";

// Main analysis page: gathers inputs and displays detector results for Layers 1–4.
import { useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { useLocale, useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import Layer3ScreenshotAnalyzer from "@/components/Layer3ScreenshotAnalyzer";
import Layer4ReferenceAnalyzer from "@/components/Layer4ReferenceAnalyzer";
import Layer4ResultCard from "@/components/Layer4ResultCard";
import type { Layer4Result } from "@/components/Layer4ResultCard";
type Comment = {
  id: string;
  text: string;
  timestamp: null;
};
// This is the optional Qwen explanation; detector scores stay in their own fields.
type FinalReport = {
  status: "generated" | "unavailable";
  summary: string;
  missing_inputs: string[];
};
// Map stable API identifiers to translation message keys.
const missingInputKeys: Record<
  string,
  "missingAuthorship" | "missingLayer3" | "missingLayer4"
> = {
  authorship: "missingAuthorship",
  layer3: "missingLayer3",
  layer4: "missingLayer4",
};
// Layer 1 can decline to classify a post, so its score and label are optional.
type AuthorshipResult = {
  status: string;
  classification?: string | null;
  probability_any_ai?: number | null;
  risk_score?: number | null;
  character_count?: number;
};

// Layer 2 returns an ad score and the cutoff used to choose its label.
type CovertAdResult = {
  label: string;
  probability: number;
  threshold: number;
};

// Layer 3 is unavailable when there are too few usable comments.
type CoordinationResult = {
  status: string;
  available: boolean;
  score: number | null;
  threshold?: number | null;
  prediction?: string | null;
  comment_count?: number;
  message?: string;
};

// The backend can be ready while still lacking enough comments to make a prediction.
function hasCoordinationPrediction(result: CoordinationResult): boolean {
  return (
    result.status === "completed" &&
    result.available &&
    typeof result.score === "number" &&
    typeof result.prediction === "string"
  );
}

// Find the first FastAPI validation path without showing the submitted content.
function rejectedInputLocation(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return null;
  }
  const detail = payload.detail;
  if (!Array.isArray(detail) || detail.length === 0) return null;
  const first = detail[0];
  if (!first || typeof first !== "object" || !("loc" in first) || !Array.isArray(first.loc)) {
    return null;
  }
  const parts = first.loc
    .filter((part: unknown): part is string | number =>
      typeof part === "string" || typeof part === "number",
    )
    .filter((part: string | number) => part !== "body");
  return (
    parts
      .map((part: string | number, index: number) =>
        typeof part === "number" ? `[${part}]` : index === 0 ? part : `.${part}`,
      )
      .join("") || null
  );
}

// Type the fields shown in result cards while retaining the full response for JSON.
type AnalyzeResponse = Record<string, unknown> & {
  authorship?: AuthorshipResult;
  covert_ad?: CovertAdResult;
  layer3?: CoordinationResult;
  layer4?: Layer4Result;
  final_report?: FinalReport;
};

// Format a 0–1 model score as a percentage without rounding away cutoff details.
function formatPercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : "—";
}

// Give the first three layers the same accessible card layout.
function ResultCard({
  title,
  outcome,
  children,
}: {
  title: string;
  outcome: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-zinc-900">{title}</h2>
      <p className="mt-3 text-base font-medium text-zinc-800">{outcome}</p>
      <div className="mt-3 space-y-2 text-sm leading-6 text-zinc-600">
        {children}
      </div>
    </section>
  );
}

// These labels cover the new cards in both languages without changing message files.
const resultLabels = {
  en: {
    authorship: "Layer 1 · AI involvement",
    aiInvolved: "AI involvement indicated",
    noAi: "AI involvement not indicated",
    covertAd: "Layer 2 · Covert advertising",
    likelyAd: "Likely advertisement",
    likelyNonAd: "Likely not an advertisement",
    coordination: "Layer 3 · Comment coordination",
    normal: "No coordination indicated",
    coordinated: "Possible coordinated comments",
    unavailable: "Insufficient data for this check",
    aiEstimate: "Estimated AI involvement",
    riskScore: "Risk score",
    modelScore: "Model score",
    cutoff: "Decision cutoff",
    characters: "Characters analyzed",
    comments: "Comments analyzed",
    invalidInput: "The backend could not accept this input.",
    errorField: "Problem field",
    checkInput: "Check this item and try again.",
    needFiveComments: "Add at least five comments to run this check.",
    authorshipHelp: "This checks writing patterns. It cannot tell which tools the author actually used.",
    adHelp: "This checks whether the post reads like a promotion. A score at or above the cutoff is flagged.",
    coordinationHelp: "This compares comments for repetition and similar wording. Few comments may not be enough to judge.",
    note: "Model predictions are estimates, not proof.",
  },
  zh: {
    authorship: "第 1 项 · 笔记文字有 AI 参与迹象吗？",
    aiInvolved: "文字可能有 AI 参与写作",
    noAi: "暂未发现明显的 AI 写作迹象",
    covertAd: "第 2 项 · 这篇笔记像推广吗？",
    likelyAd: "疑似推广内容",
    likelyNonAd: "暂未发现明显的推广迹象",
    coordination: "第 3 项 · 评论区有刷评迹象吗？",
    normal: "暂未发现明显的刷评迹象",
    coordinated: "评论可能存在集中刷评迹象",
    unavailable: "信息不够，暂时无法判断",
    aiEstimate: "AI 参与可能性（模型估计）",
    riskScore: "风险参考分（0～1，并非概率）",
    modelScore: "模型给出的分数",
    cutoff: "判定线",
    characters: "本次分析的字数",
    comments: "本次分析的评论数",
    invalidInput: "提交的内容未通过检查。",
    errorField: "出错位置",
    checkInput: "请检查这一项后再试。",
    needFiveComments: "请至少添加 5 条评论，才能查看这一项结果。",
    authorshipHelp: "系统只看文字的写法，无法知道作者实际用了什么工具。",
    adHelp: "系统会判断文字是否像推广文案；分数达到判定线时会提示疑似推广。",
    coordinationHelp: "系统会比较评论是否重复、说法是否相似；评论太少时可能无法判断。",
    note: "检测结果仅供参考，不能单独作为定论。",
  },
} as const;

// Render the input form, request analysis, and show the response on this page.
export default function Home() {
  // Load each part of the screen from the separate English/Chinese catalogs.
  const homeText = useTranslations("Home");
  const postText = useTranslations("PostText");
  const imageText = useTranslations("Images");
  const commentText = useTranslations("Comments");
  const commonText = useTranslations("Common");
  const errorText = useTranslations("Errors");
  const reportText = useTranslations("Report");
  const referenceText = useTranslations("ReferenceTool");
  // Read the active language and router so the language button can refresh UI.
  const locale = useLocale();
  const router = useRouter();
  // Pick the new card labels using the same language setting as the rest of the page.
  const resultText = locale === "zh-CN" ? resultLabels.zh : resultLabels.en;
  const [text, setText] = useState("");
  const [comments, setComments] = useState<Comment[]>([]);
  const [commentInput, setCommentInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  // This ref lets the styled upload button open the browser's native file picker.
  const imageInputRef = useRef<HTMLInputElement>(null);
  // Add one manually typed comment after validating it and assigning an ID.
  function addComment() {
    const trimmed = commentInput.trim();
    if (!trimmed) return;
    if (comments.length >= 30) {
      setError(errorText("maxComments"));
      return;
    }
    let sequence = comments.length + 1;
    let id = `c${String(sequence).padStart(2, "0")}`;
    while (comments.some((comment) => comment.id === id)) {
      sequence += 1;
      id = `c${String(sequence).padStart(2, "0")}`;
    }
    const newComment: Comment = {
      id,
      text: trimmed,
      timestamp: null,
    };
    setComments((previous) => [...previous, newComment]);
    setCommentInput("");
    setError("");
    setResult(null);
  }
  // Clean OCR output, skip duplicates, and merge it into the shared comments list.
  function addExtractedComments(extracted: string[]) {
    const cleaned = extracted
      .map((item) => item.replace(/\s+/g, " ").trim())
      .filter(Boolean);
    setComments((previous) => {
      const seen = new Set(
        previous.map((comment) => comment.text.trim().toLocaleLowerCase()),
      );
      const usedIds = new Set(previous.map((comment) => comment.id));
      const additions: Comment[] = [];
      let sequence = previous.length + 1;
      for (const value of cleaned) {
        const key = value.toLocaleLowerCase();
        if (seen.has(key) || previous.length + additions.length >= 30) continue;
        // Keep IDs unique even if the user removed an earlier comment.
        let id = `c${String(sequence).padStart(2, "0")}`;
        while (usedIds.has(id)) {
          sequence += 1;
          id = `c${String(sequence).padStart(2, "0")}`;
        }
        usedIds.add(id);
        seen.add(key);
        additions.push({ id, text: value, timestamp: null });
        sequence += 1;
      }
      return [...previous, ...additions];
    });
    setResult(null);
    setError("");
  }
  // Update one comment so the user can correct OCR mistakes before analysis.
  function editComment(id: string, text: string) {
    setComments((previous) =>
      previous.map((comment) =>
        comment.id === id ? { ...comment, text } : comment,
      ),
    );
    setResult(null);
  }
  // Remove the selected comment from the shared list.
  function removeComment(id: string) {
    setComments((previous) =>
      previous.filter((comment) => comment.id !== id),
    );
    setResult(null);
  }
  // Validate post images, then convert them to data URLs for the backend API.
  function handleImages(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    // Clear the input so choosing the same file again still triggers onChange.
    event.target.value = "";
    if (files.length > 2) {
      setError(errorText("maxImages"));
      event.target.value = "";
      return;
    }
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    if (totalBytes > 2 * 1024 * 1024) {
      setError(errorText("imagesTooLarge"));
      event.target.value = "";
      return;
    }
    setError("");
    // FileReader turns the selected images into the data URLs expected by the
    // existing /api/analyze JSON request.
    Promise.all(
      files.map(
        (file) =>
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(new Error(errorText("readImageFailed")));
            reader.readAsDataURL(file);
          }),
      ),
    )
      .then((dataUrls) => {
        setImages(dataUrls);
        setResult(null);
      })
      .catch(() => {
        setError(errorText("readImageFailed"));
    });
  }
  // Remove one image preview and exclude that image from the next analysis request.
  function removeImage(indexToRemove: number) {
    setImages((previous) => previous.filter((_, index) => index !== indexToRemove));
    setResult(null);
  }
  // Save a language choice in a cookie, then refresh so next-intl reloads messages.
  function changeLanguage(nextLocale: "en" | "zh-CN") {
    document.cookie = `locale=${nextLocale}; Path=/; Max-Age=31536000; SameSite=Lax`;
    router.refresh();
  }
  // Send post text, comments, images, and the request for a Qwen report to the API.
  async function analyze() {
    if (loading) return;
    if (!text.trim()) {
      setError(errorText("missingPostText"));
      return;
    }
    const input = {
      text,
      comments: comments.map((comment) => comment.text.trim()).filter(Boolean),
      images,
      // This opts in to the Qwen call after the four detector results are ready.
      include_report: true,
    };
    const serialized = JSON.stringify(input);
    const requestBytes = new Blob([serialized]).size;
    if (requestBytes > 3.5 * 1024 * 1024) {
      setError(errorText("requestTooLarge"));
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: serialized,
      });
    // A 422 means the backend is reachable but rejected a field in our request.
    if (response.status === 422) {
      let field: string | null = null;
      try {
        field = rejectedInputLocation(await response.json());
      } catch {
        // Keep a useful message if the server did not send a JSON error body.
      }
      setError(
        `${resultText.invalidInput}${field ? ` ${resultText.errorField}: ${field}.` : ""} ${resultText.checkInput}`,
      );
      return;
    }
    if (!response.ok) {
        setError(errorText("analysisUnavailable"));
        return;
      }
      const data = (await response.json()) as AnalyzeResponse;
      setResult(data);
    } catch {
      setError(errorText("analysisUnavailable"));
    } finally {
      setLoading(false);
    }
  }
  return (
    <main className="min-h-screen bg-zinc-100 px-6 py-10">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Page title and the language controls. */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900">
              {homeText("title")}
            </h1>
            <p className="mt-2 text-zinc-600">{homeText("tagline")}</p>
          </div>
          <div className="flex shrink-0 gap-2" aria-label={commonText("language")}>
            <button
              type="button"
              onClick={() => changeLanguage("en")}
              aria-pressed={locale === "en"}
              className="rounded border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-800 aria-pressed:border-zinc-900 aria-pressed:bg-zinc-900 aria-pressed:text-white"
            >
              English
            </button>
            <button
              type="button"
              onClick={() => changeLanguage("zh-CN")}
              aria-pressed={locale === "zh-CN"}
              className="rounded border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-800 aria-pressed:border-zinc-900 aria-pressed:bg-zinc-900 aria-pressed:text-white"
            >
              中文
            </button>
          </div>
        </div>
        {/* Post text used by Layers 1, 2, and 4. */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            {postText("title")}
          </h2>
          <textarea
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setResult(null);
            }}
            maxLength={10000}
            placeholder={postText("placeholder")}
            className="min-h-48 w-full rounded-lg border border-zinc-300 p-4 text-zinc-900 outline-none focus:border-zinc-500"
          />
          <div className="mt-2 text-right text-sm text-zinc-500">
            {postText("characterCount", { count: text.length })}
          </div>
        </section>
        {/* Post images are converted to data URLs and checked by Layer 4. */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            {imageText("title")}
          </h2>
          <p className="mb-3 text-sm text-zinc-500">
            {imageText("description")}
          </p>
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleImages}
            className="sr-only"
          />
          {/* This visible button opens the hidden file input's system picker. */}
          <button
            type="button"
            onClick={() => imageInputRef.current?.click()}
            className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            {imageText("chooseImages")}
          </button>
          {images.length > 0 && (
            <p className="mt-3 text-sm text-zinc-600">
              {imageText("selectedCount", { count: images.length })}
            </p>
          )}
          {/* Show selected post photos immediately and let the user remove each one. */}
          {images.length > 0 && (
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
              {images.map((image, index) => (
                <div key={`${index}-${image.slice(0, 32)}`} className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={image}
                    alt={imageText("previewAlt", { number: index + 1 })}
                    className="h-36 w-full rounded-lg border border-zinc-200 bg-zinc-100 object-contain"
                  />
                  <button
                    type="button"
                    onClick={() => removeImage(index)}
                    aria-label={imageText("removeImage", { number: index + 1 })}
                    className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full bg-black/75 text-xl leading-none text-white hover:bg-black focus:outline-none focus:ring-2 focus:ring-white"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
        {/* Both comment entry options feed the same list for the main analysis. */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            {commentText("title")}
          </h2>
          <p className="mb-4 text-sm leading-6 text-zinc-600">
            {commentText("instructions")}
          </p>
          <Layer3ScreenshotAnalyzer onCommentsExtracted={addExtractedComments} />
          <h3 className="mb-2 mt-6 text-sm font-semibold text-zinc-800">
            {commentText("manualOption")}
          </h3>
          <div className="flex gap-2">
            <input
              type="text"
              value={commentInput}
              onChange={(event) => setCommentInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addComment();
              }}
              placeholder={commentText("inputPlaceholder")}
              className="flex-1 rounded-lg border border-zinc-300 px-4 py-2 text-zinc-900 outline-none focus:border-zinc-500"
            />
            <button
              type="button"
              onClick={addComment}
              disabled={comments.length >= 30}
              className="rounded-lg bg-zinc-800 px-5 py-2 text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {commonText("add")}
            </button>
          </div>
          <p className="mt-2 text-sm text-zinc-500">
            {commentText("count", { count: comments.length })}
          </p>
          <div className="mt-4 space-y-2">
            {comments.map((comment) => (
              <div
                key={comment.id}
                className="flex items-center justify-between rounded-lg bg-zinc-100 p-3"
              >
                <div className="flex min-w-0 flex-1 items-center">
                  <span className="mr-2 text-xs font-semibold text-zinc-500">
                    {comment.id}
                  </span>
                  <input
                    type="text"
                    value={comment.text}
                    aria-label={commentText("editAriaLabel", { id: comment.id })}
                    onChange={(event) =>
                      editComment(comment.id, event.target.value)
                    }
                    className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 text-zinc-800 outline-none focus:border-zinc-300 focus:bg-white"
                  />
                </div>
                <button
                  type="button"
                  onClick={() => removeComment(comment.id)}
                  className="ml-4 text-sm text-red-600 hover:text-red-800"
                >
                  {commonText("remove")}
                </button>
              </div>
            ))}
          </div>
        </section>
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}
        {/* This remains the main Layer 1–4 analysis button. */}
        <button
          type="button"
          onClick={analyze}
          disabled={loading}
          className="w-full rounded-xl bg-black px-6 py-4 font-semibold text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? homeText("analyzing") : homeText("analyze")}
        </button>
        {/* Show the three text/comment detector results before the image result. */}
        {result && (
          <div className="grid gap-4 md:grid-cols-3" aria-live="polite">
            {result.authorship && (
              <ResultCard
                title={resultText.authorship}
                outcome={
                  result.authorship.status !== "supported" ||
                  !result.authorship.classification
                    ? resultText.unavailable
                    : result.authorship.classification === "ai_involved"
                      ? resultText.aiInvolved
                    : result.authorship.classification === "human_like" ||
                        result.authorship.classification === "human_written" ||
                          result.authorship.classification === "human_likely"
                        ? resultText.noAi
                        : result.authorship.classification.replaceAll("_", " ")
                }
              >
                {result.authorship.status === "supported" && (
                  <>
                    <p>{resultText.aiEstimate}: {formatPercent(result.authorship.probability_any_ai)}</p>
                    {typeof result.authorship.risk_score === "number" && (
                      <p>{resultText.riskScore}: {result.authorship.risk_score.toFixed(3)}</p>
                    )}
                    {typeof result.authorship.character_count === "number" && (
                      <p>{resultText.characters}: {result.authorship.character_count}</p>
                    )}
                  </>
                )}
                <p>{resultText.authorshipHelp}</p>
                <p>{resultText.note}</p>
              </ResultCard>
            )}
            {result.covert_ad && (
              <ResultCard
                title={resultText.covertAd}
                outcome={
                  result.covert_ad.label === "likely_ad"
                    ? resultText.likelyAd
                    : result.covert_ad.label === "likely_non_ad"
                      ? resultText.likelyNonAd
                      : result.covert_ad.label.replaceAll("_", " ")
                }
              >
                <p>{resultText.modelScore}: {formatPercent(result.covert_ad.probability)}</p>
                <p>{resultText.cutoff}: {formatPercent(result.covert_ad.threshold)}</p>
                <p>{resultText.adHelp}</p>
                <p>{resultText.note}</p>
              </ResultCard>
            )}
            {result.layer3 && (
              <ResultCard
                title={resultText.coordination}
                outcome={
                  !hasCoordinationPrediction(result.layer3)
                    ? resultText.unavailable
                    : result.layer3.prediction === "normal"
                      ? resultText.normal
                      : result.layer3.prediction === "coordinated"
                        ? resultText.coordinated
                        : (result.layer3.prediction ?? resultText.unavailable).replaceAll("_", " ")
                }
              >
                {hasCoordinationPrediction(result.layer3) ? (
                  <>
                    <p>{resultText.modelScore}: {formatPercent(result.layer3.score)}</p>
                    {typeof result.layer3.threshold === "number" && (
                      <p>{resultText.cutoff}: {formatPercent(result.layer3.threshold)}</p>
                    )}
                  </>
                ) : (
                  <p>
                    {result.layer3.status === "insufficient_data" &&
                    typeof result.layer3.comment_count === "number" &&
                    result.layer3.comment_count < 5
                      ? resultText.needFiveComments
                      : locale === "zh-CN"
                        ? resultText.unavailable
                        : (result.layer3.message ?? resultText.unavailable)}
                  </p>
                )}
                {typeof result.layer3.comment_count === "number" && (
                  <p>{resultText.comments}: {result.layer3.comment_count}</p>
                )}
                <p>{resultText.coordinationHelp}</p>
                <p>{resultText.note}</p>
              </ResultCard>
            )}
          </div>
        )}
        {/* Human-readable Layer 4 output from the main Analyze request. */}
        {result?.layer4 && <Layer4ResultCard result={result.layer4} />}
        {/* Render the optional Qwen explanation returned by /analyze. */}
        {result?.final_report && (
          <section className="rounded-xl bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-zinc-900">
              {reportText("title")}
            </h2>
            {/* The backend provides a fallback message if Qwen is unavailable. */}
            <p className="mt-3 whitespace-pre-wrap text-zinc-700">
              {result.final_report.summary}
            </p>
            {/* Tell the user which layers could not evaluate the supplied data. */}
            {result.final_report.missing_inputs.length > 0 && (
              <p className="mt-3 text-sm text-zinc-500">
                {reportText("insufficientData")}
                {result.final_report.missing_inputs
                  .map((layer) => {
                    const messageKey = missingInputKeys[layer];
                    return messageKey ? reportText(messageKey) : layer;
                  })
                  .join(reportText("separator"))}
              </p>
            )}
          </section>
        )}
        {/* Keep the full response available for development and debugging. */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-zinc-900">
            {reportText("rawResultTitle")}
          </h2>
          {result ? (
            <details className="mt-3">
              <summary className="cursor-pointer text-sm font-medium text-zinc-700">
                {reportText("showRawJson")}
              </summary>
              <pre className="mt-3 overflow-x-auto rounded-lg bg-zinc-950 p-4 text-sm text-green-400">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          ) : (
            <div className="mt-3 rounded-lg bg-zinc-100 p-4 text-zinc-500">
              {reportText("emptyState")}
            </div>
          )}
        </section>
        {/* This separate tool compares two images; it is not part of main analysis. */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-semibold text-zinc-900">
            {referenceText("title")}
          </h2>
          <p className="mb-4 text-sm leading-6 text-zinc-600">
            {referenceText("description")}
          </p>
          <Layer4ReferenceAnalyzer />
        </section>
      </div>
    </main>
  );
}
