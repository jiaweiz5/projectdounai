"use client";

import { useRef, useState } from "react";
import type { ChangeEvent, CSSProperties } from "react";
import { useLocale } from "next-intl";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AdditionalChecks from "@/components/AdditionalChecks";
import "./verifier.css";

type Analysis = {
  authorship: { status: string; classification?: string; probability_any_ai?: number | null; character_count?: number; model_version?: string };
  covert_ad?: { label: string; probability: number };
};
type Mode = "text" | "images" | "comments";
const copy = {
  zh: {
    brand: "种草鉴别助手", tagline: "读懂种草，也读懂文字背后", title: "这篇种草文案，", accent: "有 AI 参与吗？",
    intro: "粘贴笔记正文，一键查看 AI 写作迹象。配图与评论，也可以进一步核验。",
    tabs: ["文案鉴别", "配图核验", "评论分析"], input: "待检测文案", sample: "试用示例", sampleHint: "演示文案，仅用于体验操作，不代表真人样本。",
    placeholder: "把想了解的种草笔记粘贴到这里…\n\n保留原文的标点、分段和 emoji，完整的上下文有助于分析。",
    upload: "导入文本", clear: "清空", detect: "开始鉴别", busy: "正在分析…", shortcut: "Ctrl / ⌘ + Enter 开始检测",
    result: "鉴别结果", waiting: "等待检测", empty: "这篇文案，有哪些写作迹象？", emptyHelp: "在左侧粘贴一篇笔记，点击「开始鉴别」，结果会显示在这里。",
    analyzing: "正在读懂这篇文案", analyzingHelp: "正在分析文字特征，请稍候。", retry: "检测未完成", retryHelp: "请检查提示后重新检测。",
    ai: "发现 AI 参与写作的迹象", human: "暂未发现明显 AI 写作迹象", unknown: "信息不足，暂时无法判断", unknownHelp: "请提供更完整的中文笔记正文，再试一次。",
    score: "AI 参与可能性", explanation: "这是对整篇文案的模型估计，不代表文中有多少字由 AI 写成。",
    note: "结果仅供辅助判断，不能证明作者实际使用了 AI，也不代表商品或使用体验的真假。",
    download: "下载结果", ad: "补充观察 · 推广倾向", adYes: "文字存在推广倾向", adNo: "暂未发现明显推广倾向", adHelp: "推广倾向与 AI 写作是两项独立判断。",
    footer: "独立研究项目 · 非小红书官方产品", privacy: "只分析文字特征，不替你判断好物。",
    tooShort: "请至少输入 20 个字，并尽量保留完整正文。", fileError: "请导入 UTF-8 编码的 .txt 文件，正文不超过 10,000 字。",
    network: "检测服务暂时不可用，请稍后重试。", invalid: "检测服务没有返回有效结果，请重试。",
    tips: ["01  粘贴完整笔记", "02  查看写作迹象", "03  按需补充核验"], help: "如何使用", helpText: "粘贴 20–10,000 字正文后开始检测。修改文字后需重新检测；配图核验和评论分析可从上方切换。",
  },
  en: {
    brand: "NoteCheck", tagline: "A closer look at the words behind a recommendation", title: "A recommendation,", accent: "written with AI?",
    intro: "Paste a post to check for signs of AI writing. Explore images and comments when you need more context.",
    tabs: ["Text detection", "Image checks", "Comment analysis"], input: "Text to check", sample: "Try a sample", sampleHint: "Demo text for trying the interface, not a verified human sample.",
    placeholder: "Paste a recommendation here…\n\nKeep the original punctuation, paragraphs and emoji for context.",
    upload: "Import text", clear: "Clear", detect: "Check text", busy: "Analyzing…", shortcut: "Ctrl / ⌘ + Enter to check",
    result: "Detection result", waiting: "Ready when you are", empty: "What does the writing suggest?", emptyHelp: "Paste a post on the left and select Check text. Your result will appear here.",
    analyzing: "Reading your post", analyzingHelp: "Analyzing writing patterns. Please wait.", retry: "Check not completed", retryHelp: "Review the message and try again.",
    ai: "Signs of AI involvement detected", human: "No clear signs of AI involvement", unknown: "Not enough evidence", unknownHelp: "Try a more complete Chinese post.",
    score: "Estimated AI involvement", explanation: "An estimate for the whole post, not the percentage of words written by AI.",
    note: "Results are signals, not proof of AI use or of a product’s quality or authenticity.", download: "Download result",
    ad: "Additional check · Promotional tone", adYes: "Promotional tone detected", adNo: "No clear promotional tone", adHelp: "Promotion and AI authorship are separate checks.",
    footer: "Independent research project · Not affiliated with Xiaohongshu", privacy: "Writing patterns, not product verdicts.",
    tooShort: "Enter at least 20 characters, ideally the complete post.", fileError: "Import a UTF-8 .txt file with no more than 10,000 characters.",
    network: "Detection is temporarily unavailable. Please try again.", invalid: "The service returned an invalid result. Please retry.",
    tips: ["01  Paste a complete post", "02  Review writing patterns", "03  Explore extra checks"], help: "How it works", helpText: "Paste 20–10,000 characters. Recheck after editing. Switch tabs for image and comment checks.",
  },
};
const example = "最近换了一支洁面，用了一个多星期，简单记一下感受。\n\n泡沫挺细的，洗完脸颊没有紧绷感，这点我很喜欢。不过香味比预想中重，介意香味的姐妹可以先试小样。\n\n我一般只在晚上用，早上还是清水。暂时没有打算囤货，先把这一支用完再说～";

function Mark({ kind = "text" }: { kind?: Mode | "spark" }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === "text" ? <><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 12h7M9 16h5"/></> : kind === "images" ? <><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1"/><path d="m4 17 5-5 4 4 3-3 5 5"/></> : kind === "comments" ? <path d="M20 11a8 8 0 0 1-8 8H4l1-4a8 8 0 1 1 15-4ZM8 10h8M8 13h5"/> : <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z"/>}
  </svg>;
}

export default function Home() {
  const locale = useLocale();
  const t = locale === "en" ? copy.en : copy.zh;
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [sample, setSample] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Analysis | null>(null);
  const [help, setHelp] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const count = Array.from(text).length;
  const a = result?.authorship;
  const p = a?.status === "supported" && typeof a.probability_any_ai === "number" && Number.isFinite(a.probability_any_ai) && a.probability_any_ai >= 0 && a.probability_any_ai <= 1 ? a.probability_any_ai : null;
  const known = p !== null && ["ai_involved", "human_like", "human_written", "human_likely"].includes(a?.classification ?? "");
  const outcome = !known ? t.unknown : a?.classification === "ai_involved" ? t.ai : t.human;

  function updateText(value: string) { requestId.current++; setText(value); setResult(null); setError(""); setSample(false); }
  async function importFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = "";
    if (!file) return;
    const id = ++requestId.current;
    if (!file.name.toLowerCase().endsWith(".txt") || file.size > 100000) { setError(t.fileError); return; }
    try {
      const value = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
      if (requestId.current !== id) return;
      if (Array.from(value).length > 10000) { setError(t.fileError); return; }
      updateText(value);
    } catch { if (requestId.current === id) setError(t.fileError); }
  }
  async function analyze() {
    if (loading) return;
    if (Array.from(text.trim()).length < 20) { setError(t.tooShort); return; }
    const id = ++requestId.current;
    setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, comments: [], images: [], include_report: false }), signal: AbortSignal.timeout(60000) });
      if (!response.ok) throw new Error(t.network);
      const data = await response.json();
      if (!data?.authorship || typeof data.authorship.status !== "string") throw new Error(t.invalid);
      if (requestId.current === id) setResult(data);
    } catch (e) { if (requestId.current === id) setError(e instanceof Error && [t.network, t.invalid].includes(e.message) ? e.message : t.network); }
    finally { setLoading(false); }
  }
  function download() {
    if (!result) return;
    const content = `${t.brand}\n${new Date().toLocaleString(locale)}\n\n${outcome}\n${known ? `${t.score}: ${(p! * 100).toFixed(1)}%\n${t.explanation}\n` : ""}\n${t.note}\n\n${t.input}\n${text}`;
    const url = URL.createObjectURL(new Blob(["\uFEFF", content], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = "notecheck-result.txt"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <div className="verifier-shell">
    <header className="site-header"><Link className="brand" href="/"><span className="brand-icon"><Mark kind="spark"/></span>{t.brand}<span className="beta">BETA</span></Link><div className="header-actions"><button onClick={() => setHelp(!help)} aria-expanded={help}>{t.help}</button><span className="header-divider"/><button aria-label={locale === "en" ? "切换为中文" : "Switch to English"} onClick={() => { document.cookie = `locale=${locale === "en" ? "zh-CN" : "en"}; Path=/; Max-Age=31536000; SameSite=Lax`; router.refresh(); }}>{locale === "en" ? "中文" : "EN"}</button></div></header>
    {help && <div className="help-banner">{t.helpText}</div>}
    <main className="workspace">
      <section className="hero"><p className="eyebrow"><span/>{t.tagline}</p><h1>{t.title}<em>{t.accent}</em></h1><p className="hero-description">{t.intro}</p></section>
      <nav className="mode-switch" aria-label={locale === "en" ? "Check type" : "核验类型"}>{(["text", "images", "comments"] as const).map((item, i) => <button key={item} aria-pressed={mode === item} onClick={() => setMode(item)}><Mark kind={item}/>{t.tabs[i]}</button>)}</nav>
      <div hidden={mode !== "text"}>
        <div className="detection-grid">
          <section className="editor-card" aria-label={t.input}><div className="editor-heading"><h2><Mark/>{t.input}</h2><button className="sample-button" disabled={loading} onClick={() => { updateText(example); setSample(true); }}>{t.sample}<span aria-hidden="true"> ↗</span></button></div>
            <textarea aria-label={t.input} value={text} disabled={loading} maxLength={10000} placeholder={t.placeholder} onChange={e => updateText(e.target.value)} onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); void analyze(); } }}/>
            {sample && <p className="sample-note">{t.sampleHint}</p>}
            {error && <p className="error-message" role="alert">{error}</p>}
            <div className="editor-meta"><span>{t.shortcut}</span><span>{count.toLocaleString()} / 10,000</span></div>
            <div className="editor-actions"><input aria-label={t.upload} ref={fileInput} type="file" accept=".txt,text/plain" hidden onChange={importFile}/><button className="import-button" disabled={loading} onClick={() => fileInput.current?.click()}><span aria-hidden="true">↑</span>{t.upload}</button><div><button className="clear-button" disabled={loading || !text} onClick={() => updateText("")}>{t.clear}</button><button className="primary-button" disabled={loading || !text.trim()} onClick={analyze}>{loading ? t.busy : t.detect}<span aria-hidden="true"> →</span></button></div></div>
          </section>
          <aside className="result-panel" aria-live="polite" aria-busy={loading}><div className="result-heading"><h2>{t.result}</h2><span className="status-label">{loading ? t.busy : result ? "✓" : t.waiting}</span></div>
            {!result ? <div className="result-empty"><div className={`empty-orbit ${loading ? "is-loading" : ""}`}><Mark kind="spark"/></div><h3>{loading ? t.analyzing : error ? t.retry : t.empty}</h3><p>{loading ? t.analyzingHelp : error ? t.retryHelp : t.emptyHelp}</p><div className="placeholder-lines" aria-hidden="true"><span/><span/><span/></div></div> : <div className="result-content"><p className={`outcome ${known && a?.classification !== "ai_involved" ? "human" : ""}`}>{outcome}</p>{known ? <><div className="score-ring" style={{ "--score": `${p! * 100}%` } as CSSProperties}><div><strong>{(p! * 100).toFixed(1)}<small>%</small></strong><span>{t.score}</span></div></div><p className="score-explanation">{t.explanation}</p></> : <p className="score-explanation unknown-copy">{t.unknownHelp}</p>}
              {result.covert_ad && ["likely_ad", "likely_non_ad"].includes(result.covert_ad.label) && <details className="ad-detail"><summary>{t.ad}</summary><p>{result.covert_ad.label === "likely_ad" ? t.adYes : t.adNo}</p><p>{t.adHelp}</p></details>}<button className="download-button" onClick={download}>↓ {t.download}</button></div>}
            <p className="result-note">ⓘ {t.note}</p>
          </aside>
        </div>
        <div className="steps">{t.tips.map(tip => <span key={tip}>{tip}</span>)}</div>
      </div>
      <div hidden={mode !== "images"}><AdditionalChecks mode="images"/></div>
      <div hidden={mode !== "comments"}><AdditionalChecks mode="comments"/></div>
      <footer className="site-footer"><span>{t.footer}</span><span>{t.privacy}</span></footer>
    </main>
  </div>;
}
