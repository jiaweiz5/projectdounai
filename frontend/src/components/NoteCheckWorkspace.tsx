"use client";

import { useRef, useState } from "react";
import { useLocale } from "next-intl";
import type { Layer4Result } from "./Layer4ResultCard";
import styles from "./NoteCheckWorkspace.module.css";

type Photo = { file: File; data: string; original?: { file: File; data: string } };
type Comment = { id: number; text: string };
type MainResult = {
  covert_ad?: { label: string; probability?: number };
  layer3?: { status: string; available: boolean; prediction?: string; score?: number; comment_count?: number };
  layer4?: Layer4Result;
};
type ReferenceResult = { status: string; available: boolean; label?: number; edit_probability?: number; changed_region_count?: number };
type Findings = { main: MainResult | null; mainFailed: boolean; references: { index: number; data: ReferenceResult | null }[] };
type OCRLine = { text: string; confidence?: number };

function Icon({ kind }: { kind: "text" | "image" | "comments" | "check" }) {
  return <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === "text" ? <><path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h5"/></> : kind === "image" ? <><rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8" cy="8" r="1.4"/><path d="m3 17 5-5 4 4 4-5 5 6"/></> : kind === "comments" ? <path d="M20 11a8 8 0 0 1-8 8H4l1-4a8 8 0 1 1 15-4ZM8 10h8M8 13h5"/> : <path d="m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7zM8 12l3 3 5-6"/>}
  </svg>;
}

async function readImage(file: File, limit: number): Promise<string> {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > limit) throw new Error("image");
  const data = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file);
  });
  const img = new Image(); img.src = data; await img.decode();
  return data;
}
const validNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const validProbability = (value: unknown): value is number => validNumber(value) && value >= 0 && value <= 1;

export default function NoteCheckWorkspace() {
  const en = useLocale() === "en";
  const l = (zh: string, eng: string) => en ? eng : zh;
  const [text, setText] = useState("");
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [comments, setComments] = useState<Comment[]>([]);
  const [draft, setDraft] = useState("");
  const [screenshot, setScreenshot] = useState<{ file: File; data: string } | null>(null);
  const [ocrLines, setOcrLines] = useState<OCRLine[]>([]);
  const [ocrMessage, setOcrMessage] = useState("");
  const [imported, setImported] = useState(false);
  const [reviewed, setReviewed] = useState(true);
  const [busy, setBusy] = useState<"" | "reading" | "ocr" | "analysis">("");
  const [error, setError] = useState("");
  const [findings, setFindings] = useState<Findings | null>(null);
  const sequence = useRef(0);
  const inFlight = useRef(false);
  const photoInput = useRef<HTMLInputElement>(null);
  const screenInput = useRef<HTMLInputElement>(null);
  const usable = comments.filter(c => c.text.trim());
  const changed = () => { setFindings(null); setError(""); };

  async function choose(files: File[], kind: "photos" | "screenshot" | "original", index = 0) {
    if (!files.length || inFlight.current) return;
    changed(); inFlight.current = true; setBusy("reading");
    try {
      if (kind === "photos") {
        if (photos.length + files.length > 2 || [...photos.map(p => p.file), ...files].reduce((n, f) => n + f.size, 0) > 2 * 1024 * 1024) throw new Error("photos");
        const next = await Promise.all(files.map(async file => ({ file, data: await readImage(file, 2 * 1024 * 1024) })));
        setPhotos(prev => [...prev, ...next]);
      } else {
        const file = files[0]; const data = await readImage(file, 12 * 1024 * 1024);
        if (kind === "screenshot") { setScreenshot({ file, data }); setImported(false); setOcrMessage(""); setOcrLines([]); }
        else setPhotos(prev => prev.map((p, i) => i === index ? { ...p, original: { file, data } } : p));
      }
    } catch { setError(kind === "photos" ? l("配图最多 2 张、合计不超过 2 MB，请选择有效的 JPG、PNG 或 WebP。", "Choose up to 2 valid JPG, PNG or WebP images, 2 MB total.") : l("请选择可打开的 JPG、PNG 或 WebP，单张不超过 12 MB。", "Choose a valid JPG, PNG or WebP image up to 12 MB.")); }
    finally { inFlight.current = false; setBusy(""); }
  }

  function appendComments(values: string[]) {
    const clean = values.map(v => v.trim()).filter(Boolean);
    const room = 30 - comments.length;
    // Repeated comments are evidence. Never deduplicate their text.
    setComments(prev => [...prev, ...clean.slice(0, room).map(value => ({ id: ++sequence.current, text: value }))]);
    changed();
    if (clean.length > room) setError(l(`最多保留 30 条，另有 ${clean.length - room} 条未加入，请从识别原文检查。`, `Only 30 comments can be kept. ${clean.length - room} were not added; review the OCR lines.`));
  }

  async function extract() {
    if (!screenshot || inFlight.current || imported) return;
    inFlight.current = true; setBusy("ocr"); changed(); setOcrMessage("");
    try {
      const body = new FormData(); body.append("file", screenshot.file);
      const response = await fetch("/api/analyze-screenshot", { method: "POST", body, signal: AbortSignal.timeout(110000) });
      if (!response.ok) throw new Error("ocr");
      const payload = await response.json();
      if (!Array.isArray(payload?.ocr?.comments) || !Array.isArray(payload?.ocr?.lines)) throw new Error("schema");
      const values: string[] = payload.ocr.comments.map((row: OCRLine) => row.text).filter((v: unknown): v is string => typeof v === "string" && !!v.trim());
      setOcrLines(payload.ocr.lines.filter((row: OCRLine) => typeof row.text === "string"));
      appendComments(values); setReviewed(false); setImported(true);
      setOcrMessage(l(`已整理 ${values.length} 条候选评论。请检查用户名、日期和换行；漏掉的文字可从识别原文补充。`, `${values.length} candidate comments extracted. Review names, dates and wrapping; add missing text from OCR lines.`));
    } catch { setError(l("评论截图识别未完成。请重试，或手动添加评论；正文和配图仍可单独核验。", "Screenshot OCR did not complete. Retry or add comments manually; text and images can still be checked.")); }
    finally { inFlight.current = false; setBusy(""); }
  }

  async function analyze() {
    if (inFlight.current) return;
    if (!text.trim()) { setError(l("请先填写笔记正文。", "Enter the post text first.")); return; }
    if (usable.length && !reviewed) { setError(l("请检查并确认识别后的评论列表。", "Review and confirm the extracted comments first.")); return; }
    inFlight.current = true; setBusy("analysis"); changed();
    try {
      const requestedRefs = photos.flatMap((p, index) => p.original ? [{ photo: p, index }] : []);
      const tasks = await Promise.allSettled([
        (async () => {
          const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, images: photos.map(p => p.data), comments: usable.map(c => c.text.trim()), include_report: false }), signal: AbortSignal.timeout(110000) });
          if (!response.ok) throw new Error("main");
          const data = await response.json();
          if (!data || typeof data !== "object" || !("covert_ad" in data || "layer3" in data || "layer4" in data)) throw new Error("schema");
          return data as MainResult;
        })(),
        ...requestedRefs.map(async ({ photo }) => {
          const body = new FormData(); body.append("file", photo.file); body.append("reference_file", photo.original!.file);
          const response = await fetch("/api/analyze-layer4-reference", { method: "POST", body, signal: AbortSignal.timeout(110000) });
          if (!response.ok) throw new Error("reference");
          const payload = await response.json();
          if (typeof payload?.layer4_reference?.status !== "string") throw new Error("schema");
          return payload.layer4_reference as ReferenceResult;
        }),
      ]);
      const main = tasks[0];
      setFindings({ main: main.status === "fulfilled" ? main.value as MainResult : null, mainFailed: main.status === "rejected", references: requestedRefs.map(({ index }, i) => ({ index, data: tasks[i + 1].status === "fulfilled" ? (tasks[i + 1] as PromiseFulfilledResult<ReferenceResult>).value : null })) });
    } finally { inFlight.current = false; setBusy(""); }
  }

  const ad = findings?.main?.covert_ad;
  const adValid = ad && ["likely_ad", "likely_non_ad"].includes(ad.label) && validProbability(ad.probability);
  const adTitle = !adValid ? l("推广分析未完成", "Promotion analysis unavailable") : ad.label === "likely_ad" ? l("文案存在推广倾向", "Promotional language indicated") : l("暂未发现明显推广倾向", "No clear promotional language indicated");
  const group = findings?.main?.layer3;
  const groupValid = group?.status === "completed" && group.available && ["normal", "coordinated"].includes(group.prediction ?? "") && validProbability(group.score);
  const groupTitle = !usable.length ? l("未提供评论", "No comments provided") : usable.length < 5 ? l("评论不足，暂不判断", "Not enough comments") : !groupValid ? l("评论分析未完成", "Comment analysis unavailable") : group.prediction === "coordinated" ? l("发现相似或重复话术线索", "Similar or repeated talking points detected") : l("暂未发现明显协同线索", "No clear coordination indicated");

  function download() {
    if (!findings) return;
    const body = { post: text, comments: usable.map(c => c.text.trim()), images: photos.map(p => p.file.name), findings };
    const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: "application/json;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = "promotion-review.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="detection-grid">
    <div className="editor-card">
      <fieldset disabled={!!busy} className={styles.fields}>
        <section className={styles.mainSection}>
          <div className="editor-heading"><h2><Icon kind="text"/>{l("笔记正文", "Post text")}</h2><span className="sample-button">{l("必填", "Required")}</span></div>
          <textarea className="main-editor" aria-label={l("笔记正文", "Post text")} maxLength={10000} value={text} placeholder={l("把想核验的小红书笔记粘贴到这里…\n\n保留原文里的使用体验、产品名称与推广披露，让线索更完整。", "Paste the post you want to review…\n\nKeep product names, experiences and sponsorship disclosures for context.")} onChange={e => { setText(e.target.value); changed(); }}/>
          <div className="editor-meta"><span>{l("正文必填，配图和评论选填", "Post text required; images and comments optional")}</span><span>{text.length.toLocaleString()} / 10,000</span></div>
        </section>

        <details className={styles.evidence}>
          <summary className={styles.heading}><h2><Icon kind="image"/>{l("笔记配图", "Post images")}</h2><span>{photos.length ? `${photos.length} / 2` : l("选填 · 展开", "Optional · Expand")}</span></summary>
          <p className={styles.hint}>{l("比较配图与上面的正文，并识别图中文字。最多 2 张，合计 2 MB。", "Compare images with the post and extract visible text. Up to 2 images, 2 MB total.")}</p>
          <input ref={photoInput} type="file" hidden multiple accept="image/jpeg,image/png,image/webp" onChange={e => { void choose(Array.from(e.target.files ?? []), "photos"); e.target.value = ""; }}/>
          <button className={styles.upload} disabled={photos.length >= 2} onClick={() => photoInput.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); void choose(Array.from(e.dataTransfer.files), "photos"); }}><Icon kind="image"/>{l("添加笔记配图", "Add post images")}<small>JPG / PNG / WebP</small></button>
          {photos.map((photo, index) => <div className={styles.photo} key={`${photo.file.name}-${index}`}>
            <div className={styles.photoTop}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={photo.data} alt={l(`笔记配图 ${index + 1}`, `Post image ${index + 1}`)}/>
              <div><strong>{l(`配图 ${index + 1}`, `Image ${index + 1}`)}</strong><p>{photo.file.name}</p><button className={styles.link} onClick={() => { setPhotos(prev => prev.filter((_, i) => i !== index)); changed(); }}>{l("移除配图", "Remove image")}</button></div>
            </div>
            <details className={styles.reference}><summary>{l("添加可信原图，检查图片改动", "Add a trusted original to check for edits")}</summary>
              <p className={styles.hint}>{l("选择同一画面的可信原图，单张最多 12 MB。会和这张配图单独比较。", "Choose a trusted original of the same scene, up to 12 MB. It will be compared with this image.")}</p>
              <label className={styles.fileLabel}>{l("选择可信原图", "Choose original")}<input type="file" accept="image/jpeg,image/png,image/webp" onChange={e => { void choose(Array.from(e.target.files ?? []), "original", index); e.target.value = ""; }}/></label>
              {photo.original && <div className={styles.original}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={photo.original.data} alt={l("可信原图", "Trusted original")}/><span>{photo.original.file.name}</span><button className={styles.link} onClick={() => { setPhotos(prev => prev.map((p, i) => i === index ? { ...p, original: undefined } : p)); changed(); }}>{l("移除原图", "Remove original")}</button>
              </div>}
            </details>
          </div>)}
        </details>

        <details className={styles.evidence}>
          <summary className={styles.heading}><h2><Icon kind="comments"/>{l("评论区", "Comments")}</h2><span>{comments.length ? `${usable.length} / 30` : l("选填 · 展开", "Optional · Expand")}</span></summary>
          <p className={styles.hint}>{l("上传清晰的评论区截图，或手动添加。保留真实重复评论，至少 5 条才能分析协同风险。", "Upload a clear comment screenshot or type comments. Keep genuine repetitions. At least 5 comments are needed.")}</p>
          <input ref={screenInput} type="file" hidden accept="image/jpeg,image/png,image/webp" onChange={e => { void choose(Array.from(e.target.files ?? []), "screenshot"); e.target.value = ""; }}/>
          <div className={styles.row}><button className={styles.secondary} onClick={() => screenInput.current?.click()}>{l("选择评论截图", "Choose screenshot")}</button><span className={styles.hint}>JPG / PNG / WebP · ≤ 12 MB</span></div>
          {screenshot && <div className={styles.screenshot}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={screenshot.data} alt={l("评论截图预览", "Comment screenshot preview")}/>
            <div className={styles.row}><button className={styles.secondary} disabled={imported || comments.length >= 30} onClick={extract}>{busy === "ocr" ? l("正在识别…", "Reading…") : imported ? l("已加入列表", "Added to list") : l("识别并整理评论", "Extract comments")}</button><button className={styles.link} onClick={() => { setScreenshot(null); setOcrMessage(""); setOcrLines([]); setImported(false); }}>{l("移除截图", "Remove screenshot")}</button></div>
          </div>}
          {ocrMessage && <p className={styles.notice} role="status">{ocrMessage}</p>}
          {!!ocrLines.length && <details className={styles.raw}><summary>{l("查看识别原文，补充遗漏评论", "Review OCR lines and add missing comments")}</summary><p className={styles.hint}>{l("这里包含用户名、日期和界面文字。只添加属于评论正文的内容，再在列表里合并换行。", "These lines may include usernames, dates and controls. Add only comment text, then merge wrapped lines in the list.")}</p>{ocrLines.map((line, i) => <div className={styles.rawLine} key={i}><span>{line.text}</span><button disabled={comments.length >= 30} className={styles.link} onClick={() => { appendComments([line.text]); setReviewed(false); }}>{l("添加", "Add")}</button></div>)}</details>}
          <div className={styles.manual}><label htmlFor="comment-draft">{l("手动添加一条完整评论", "Add one complete comment")}</label><textarea id="comment-draft" maxLength={5000} value={draft} onChange={e => setDraft(e.target.value)} placeholder={l("多行文字会作为同一条评论保留", "Multiple lines will stay in one comment")}/><button className={styles.secondary} disabled={!draft.trim() || comments.length >= 30} onClick={() => { appendComments([draft]); setDraft(""); }}>{l("添加评论", "Add comment")}</button></div>
          <div className={styles.listHeading}><strong>{l("待分析评论", "Comments to analyze")}</strong><span>{usable.length} / 30</span></div>
          {comments.map((comment, i) => <div className={styles.comment} key={comment.id}><span>{String(i + 1).padStart(2, "0")}</span><textarea aria-label={l(`评论 ${i + 1}`, `Comment ${i + 1}`)} value={comment.text} maxLength={5000} onChange={e => { setComments(prev => prev.map(c => c.id === comment.id ? { ...c, text: e.target.value } : c)); changed(); }}/><button className={styles.link} onClick={() => { setComments(prev => prev.filter(c => c.id !== comment.id)); changed(); }}>{l("删除", "Delete")}</button></div>)}
          {usable.length > 0 && usable.length < 5 && <p className={styles.notice}>{l(`还需 ${5 - usable.length} 条评论才能分析协同风险，其他项目可以先核验。`, `Add ${5 - usable.length} more comments for coordination analysis. Other checks can run now.`)}</p>}
          {!!comments.length && <label className={styles.confirm}><input type="checkbox" checked={reviewed} onChange={e => setReviewed(e.target.checked)}/>{l("我已检查评论分段和文字，列表可以用于分析", "I have reviewed the comment text and grouping")}</label>}
        </details>
      </fieldset>
      {error && <p className="error-message" role="alert">{error}</p>}
      <div className={`editor-actions ${styles.actions}`}><span>{l("选填项不影响正文核验", "Optional evidence can be added later")}</span><button className="primary-button" disabled={!!busy || !text.trim() || (!!usable.length && !reviewed)} onClick={analyze}>{busy === "analysis" ? l("正在核验…", "Reviewing…") : busy ? l("正在处理材料…", "Processing…") : l("开始核验", "Review post")}<span aria-hidden="true"> →</span></button></div>
    </div>

    <aside className="result-panel" aria-live="polite" aria-busy={busy === "analysis"}>
      <div className="result-heading"><h2>{l("推广线索报告", "Promotion clues")}</h2><span className="status-label">{busy === "analysis" ? l("核验中", "Reviewing") : findings ? l("本次结果", "Results") : l("等待核验", "Ready")}</span></div>
      {!findings ? <><div className="result-empty"><span className="empty-orbit"><Icon kind="check"/></span><h3>{busy === "analysis" ? l("正在逐项查看线索", "Reviewing the evidence") : l("多一点材料，多一份依据", "More context, more evidence")}</h3><p>{l("从正文开始，按需添加配图与评论。每一项都有独立的结果。", "Start with the post and optionally add images and comments. Each check has its own finding.")}</p></div><div className={styles.checklist}>{[l("推广表达", "Promotional language"), l("评论协同", "Comment coordination"), l("图文一致性", "Image-text alignment"), l("原图差异", "Original comparison")].map((v, i) => <div key={v}><span>{String(i + 1).padStart(2, "0")}</span>{v}<small>{i === 0 ? l("根据正文", "Post text") : l("按需补充", "Optional")}</small></div>)}</div></> : <>
        {findings.mainFailed && <p className={styles.notice}>{l("正文、评论及图文分析未完成，请重试。已完成的原图比对仍可查看。", "Post, comment and alignment analysis failed. Retry; any completed original comparisons remain below.")}</p>}
        <section className={`${styles.finding} ${styles.primaryFinding}`}><small>{l("01 · 推广表达", "01 · Promotional language")}</small><h3>{adTitle}</h3>{adValid && <p>{l("文案推广倾向 · 模型估计", "Text promotion estimate")} <strong>{(ad.probability! * 100).toFixed(1)}%</strong></p>}<p>{l("这项结果只来自正文。评论和图片线索分别列出，不合并成推广概率。", "This finding is based on the post text. Comments and images are reported separately, not combined into a probability.")}</p></section>
        <section className={styles.finding}><small>{l("02 · 评论协同", "02 · Comment coordination")}</small><h3>{groupTitle}</h3><p>{l("本次分析的评论数", "Comments analyzed")}: {group?.comment_count ?? (usable.length ? l("未完成", "Unavailable") : 0)}</p>{groupValid && <p>{l("协同风险 · 模型估计", "Estimated coordination risk")}: {(group.score! * 100).toFixed(1)}%</p>}<p>{l("相似话术是复核线索，不能直接证明水军或商业合作。", "Similar wording alone does not prove bots or sponsorship.")}</p></section>
        <section className={styles.finding}><small>{l("03 · 图文一致性", "03 · Image-text alignment")}</small>{!photos.length ? <h3>{l("未提供配图", "No images provided")}</h3> : photos.map((_, index) => {
          const item = findings.main?.layer4?.images?.find(i => i.index === index + 1);
          const alignment = item?.alignment;
          const complete = alignment?.status === "completed" && alignment.available && [0, 1].includes(alignment.label ?? -1);
          return <div key={index} className={styles.imageFinding}><h3>{l(`配图 ${index + 1}：`, `Image ${index + 1}: `)}{alignment?.status === "provisional" ? l("需结合全文复核", "Review full caption") : !complete ? l("核验未完成", "Check unavailable") : alignment.label === 1 ? l("可能与正文不一致", "Possible mismatch") : l("与正文基本对应", "Appears aligned")}</h3>{validNumber(alignment?.image_text_similarity) && <p>{l("匹配分数（非概率）", "Match score (not probability)")}: {alignment.image_text_similarity.toFixed(4)}</p>}<div className={styles.ocrText}><strong>{l("图片中识别到的文字", "Text found in image")}</strong><p>{!item?.ocr || item.ocr.message ? l("文字识别未完成", "OCR unavailable") : item?.ocr?.ocr_text || l("未识别到文字", "No text found")}</p></div></div>;
        })}</section>
        <section className={styles.finding}><small>{l("04 · 原图差异", "04 · Original comparison")}</small>{!findings.references.length ? <h3>{l("未提供可信原图", "No trusted original provided")}</h3> : findings.references.map(({ index, data }) => {
          const complete = data?.status === "completed" && data.available && [0, 1].includes(data.label ?? -1);
          return <div key={index}><h3>{l(`配图 ${index + 1}：`, `Image ${index + 1}: `)}{!complete ? l("比对未完成", "Comparison unavailable") : data.label === 1 ? l("发现可能的改动", "Possible editing") : l("暂未发现明显改动", "No clear editing")}</h3>{complete && validProbability(data.edit_probability) && <p>{l("改动可能性 · 模型估计", "Estimated editing likelihood")}: {(data.edit_probability * 100).toFixed(1)}%</p>}{complete && Number.isInteger(data.changed_region_count) && <p>{l("差异区域", "Changed regions")}: {data.changed_region_count}</p>}</div>;
        })}<p>{l("裁剪、滤镜和压缩也会产生差异，请结合原图来源复核。", "Cropping, filters and compression can also cause differences. Review the original's source.")}</p></section>
        <button className="download-button" onClick={download}>{l("下载完整结果", "Download full results")} ↓</button>
      </>}
      <p className="result-note">{l("辅助发现线索，不直接证明商业合作或内容造假。未提供材料与未完成检测都不代表零风险。", "Findings support review, not proof of sponsorship or deception. Missing or unavailable checks do not mean zero risk.")}</p>
    </aside>
  </div>;
}
