"use client";

import { useEffect, useRef, useState } from "react";
import { useLocale } from "next-intl";
import type { Layer4Result } from "./Layer4ResultCard";
import { imageCheckCopy } from "./image-check-copy";
import styles from "./ImageCheckWorkspace.module.css";

type Method = "alignment" | "reference";
type SelectedImage = { file: File; url: string };
type ReferenceResult = {
  status: string; available: boolean; label?: number | null;
  edit_probability?: number; changed_region_count?: number;
};
type Result = { method: "alignment"; data: Layer4Result } | { method: "reference"; data: ReferenceResult };

function PictureIcon() {
  return <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8" cy="8" r="1.4"/><path d="m3 17 5-5 4 4 4-5 5 6"/></svg>;
}

function usePreview(image: SelectedImage | null) {
  useEffect(() => {
    if (!image) return;
    return () => URL.revokeObjectURL(image.url);
  }, [image]);
}

function toDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function ImageCheckWorkspace() {
  const locale = useLocale();
  const t = locale === "en" ? imageCheckCopy.en : imageCheckCopy.zh;
  const [method, setMethod] = useState<Method>("alignment");
  const [target, setTarget] = useState<SelectedImage | null>(null);
  const [original, setOriginal] = useState<SelectedImage | null>(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [reading, setReading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const targetInput = useRef<HTMLInputElement>(null);
  const originalInput = useRef<HTMLInputElement>(null);
  const selectionId = useRef(0);
  const busy = loading || reading;
  usePreview(target);
  usePreview(original);

  async function selectFiles(files: File[], kind: "target" | "original") {
    if (busy || !files.length) return;
    setResult(null); setError("");
    if (files.length !== 1) { setError(t.many); return; }
    const file = files[0];
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { setError(t.invalid); return; }
    if (file.size > 12 * 1024 * 1024) { setError(t.large); return; }
    const id = ++selectionId.current;
    const url = URL.createObjectURL(file);
    setReading(true);
    try {
      const image = new Image(); image.src = url;
      await image.decode();
      if (selectionId.current !== id) { URL.revokeObjectURL(url); return; }
      (kind === "target" ? setTarget : setOriginal)({ file, url });
    } catch { URL.revokeObjectURL(url); setError(t.invalid); }
    finally { setReading(false); }
  }

  function reset() {
    selectionId.current++;
    setTarget(null); setOriginal(null); setCaption(""); setResult(null); setError("");
  }

  async function analyze() {
    if (busy) return;
    setError(""); setResult(null);
    if (!target) { setError(t.needImage); return; }
    if (method === "reference" && !original) { setError(t.needReference); return; }
    if (method === "alignment" && !caption.trim()) { setError(t.needCaption); return; }
    if (method === "alignment" && target.file.size > 2 * 1024 * 1024) { setError(t.alignmentLarge); return; }
    if (caption.length > 10000) { setError(t.captionLong); return; }
    setLoading(true);
    try {
      let response: Response;
      if (method === "reference" && original) {
        const body = new FormData(); body.append("file", target.file); body.append("reference_file", original.file);
        response = await fetch("/api/analyze-layer4-reference", { method: "POST", body, signal: AbortSignal.timeout(110000) });
      } else {
        const image = await toDataUrl(target.file);
        response = await fetch("/api/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: caption, images: [image], comments: [], include_report: false }),
          signal: AbortSignal.timeout(110000),
        });
      }
      if (!response.ok) throw new Error(t.network);
      const payload = await response.json();
      const data = method === "reference" ? payload?.layer4_reference : payload?.layer4;
      if (!data || typeof data.status !== "string") throw new Error(t.network);
      setResult(method === "reference" ? { method, data: data as ReferenceResult } : { method, data: data as Layer4Result });
    } catch (e) {
      setError(e instanceof Error && e.name === "TimeoutError" ? t.timeout : t.network);
    } finally { setLoading(false); }
  }

  const complete = result?.data.status === "completed" && result.data.available === true && [0, 1].includes(result.data.label ?? -1);
  const provisional = result?.data.status === "provisional";
  const concern = complete && result?.data.label === 1;
  const verdict = provisional ? t.provisional : !complete ? t.unavailable : result?.method === "reference" ? (concern ? t.edited : t.unchanged) : (concern ? t.mismatch : t.aligned);
  const explanation = provisional ? t.provisionalHint : !complete ? t.unavailableHint : result?.method === "reference" ? t.referenceResultHint : t.alignedHint;
  const alignment = result?.method === "alignment" ? result.data.images?.[0] : undefined;
  const score = result?.method === "reference" ? result.data.edit_probability : alignment?.alignment.image_text_similarity;
  const validScore = complete && typeof score === "number" && Number.isFinite(score) && (result?.method !== "reference" || (score >= 0 && score <= 1));

  function download() {
    if (!result || !target) return;
    const report = [t.result, new Date().toLocaleString(locale), t[method], target.file.name, method === "reference" ? original?.file.name : caption, verdict, explanation,
      validScore ? `${result.method === "reference" ? t.probability : t.matchScore}: ${result.method === "reference" ? `${(score! * 100).toFixed(1)}%` : score!.toFixed(4)}` : "", t.note].filter(Boolean).join("\n\n");
    const url = URL.createObjectURL(new Blob(["\uFEFF", report], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = "image-check-result.txt"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <>
    <div className="detection-grid">
      <section className={`editor-card ${styles.editor}`} aria-label={t.title}>
        <div className="editor-heading"><h2><PictureIcon/>{t.title}</h2><div className={styles.methods} aria-label={locale === "en" ? "Image check method" : "图片核验方式"}>
          {(["alignment", "reference"] as const).map(value => <button key={value} disabled={busy} aria-pressed={method === value} onClick={() => { setMethod(value); setResult(null); setError(""); }}>{t[value]}</button>)}
        </div></div>
        <p className={styles.methodHint}>{method === "alignment" ? t.alignmentHint : t.referenceHint}</p>
        <input ref={targetInput} type="file" accept="image/jpeg,image/png,image/webp" aria-label={t.choose} hidden disabled={busy} onChange={e => { const files = Array.from(e.target.files ?? []); e.target.value = ""; void selectFiles(files, "target"); }}/>
        <button type="button" disabled={busy} className={`${styles.dropzone} ${dragging ? styles.dragging : ""} ${target ? styles.withImage : ""}`} aria-label={target ? t.replace : t.choose}
          onClick={() => targetInput.current?.click()}
          onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); void selectFiles(Array.from(e.dataTransfer.files), "target"); }}
          onPaste={e => { if (e.clipboardData.files.length) { e.preventDefault(); void selectFiles(Array.from(e.clipboardData.files), "target"); } }}>
          {target ? <>
            {/* Local object URL; keep full image visible without cropping. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={target.url} alt={t.title}/><span className={styles.replaceOverlay}>{t.replace} ↗</span>
          </> : <><span className={styles.uploadIcon}><PictureIcon/><span>＋</span></span><strong>{t.choose}</strong><span>{t.drop}</span><small>{t.formats}</small></>}
        </button>
        {target && <div className={styles.fileRow}><span title={target.file.name}>{target.file.name}</span><small>{(target.file.size / 1024 / 1024).toFixed(2)} MB</small><button disabled={busy} onClick={() => { setTarget(null); setResult(null); setError(""); }}>{t.remove}</button></div>}
        {method === "alignment" ? <div className={styles.caption}><label htmlFor="image-caption">{t.caption}</label><textarea id="image-caption" disabled={busy} value={caption} maxLength={10000} placeholder={t.placeholder} onChange={e => { setCaption(e.target.value); setResult(null); setError(""); }}/><span>{caption.length.toLocaleString()} / 10,000</span></div> : <div className={styles.originalRow}>
          <input ref={originalInput} type="file" accept="image/jpeg,image/png,image/webp" aria-label={t.uploadOriginal} hidden disabled={busy} onChange={e => { const files = Array.from(e.target.files ?? []); e.target.value = ""; void selectFiles(files, "original"); }}/>
          <button className={styles.originalPreview} disabled={busy} aria-label={t.uploadOriginal} onClick={() => originalInput.current?.click()}>{original ? <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={original.url} alt={t.original}/></> : <PictureIcon/>}</button>
          <div><strong>{t.original}</strong><p title={original?.file.name}>{original?.file.name ?? t.originalHint}</p></div>
          <button className="sample-button" disabled={busy} onClick={() => originalInput.current?.click()}>{original ? t.replace : t.uploadOriginal}</button>
        </div>}
        {error && <p className="error-message" role="alert">{error}</p>}
        <div className={`editor-actions ${styles.actions}`}><span>{method === "alignment" ? "JPG / PNG / WebP · ≤ 2 MB" : "JPG / PNG / WebP · ≤ 12 MB"}</span><div><button className="clear-button" disabled={busy || (!target && !original && !caption)} onClick={reset}>{t.clear}</button><button className="primary-button" disabled={busy || !target || (method === "reference" ? !original : !caption.trim())} onClick={analyze}>{busy ? t.busy : t.start}<span aria-hidden="true"> →</span></button></div></div>
      </section>
      <aside className={`result-panel ${styles.result}`} aria-live="polite" aria-busy={loading}>
        <div className="result-heading"><h2>{t.result}</h2><span className="status-label">{loading ? t.busy : result ? t[method] : t.waiting}</span></div>
        {result ? <div className="result-content"><p className={`outcome ${complete && !concern ? "human" : ""} ${!complete ? styles.neutral : ""}`}>{verdict}</p>
          {validScore && <div className={styles.score}><span>{result.method === "reference" ? t.probability : t.matchScore}</span><strong>{result.method === "reference" ? (score! * 100).toFixed(1) : score!.toFixed(4)}{result.method === "reference" && <small>%</small>}</strong></div>}
          <p className={`score-explanation ${styles.explanation}`}>{explanation}</p>
          {validScore && result.method === "alignment" && <p className="score-explanation">{t.matchHelp}</p>}
          {complete && result.method === "reference" && Number.isInteger(result.data.changed_region_count) && <p className={styles.regions}>{t.regions}<strong>{result.data.changed_region_count}</strong></p>}
          {alignment?.ocr && <div className={styles.ocrCard}><div><strong>{t.ocr}</strong><span>{alignment.ocr.ocr_item_count} · {Math.round((alignment.ocr.ocr_confidence || 0) * 100)}%</span></div><p className={styles.ocr}>{alignment.ocr.ocr_text || t.noOcr}</p></div>}
          <button className="download-button" onClick={download}>↓ {t.download}</button>
        </div> : <div className="result-empty"><div className={`empty-orbit ${loading ? "is-loading" : ""}`}><PictureIcon/></div><h3>{loading ? t.loading : error ? t.failed : t.empty}</h3><p>{loading ? t.loadingHint : error ? t.failedHint : t.emptyHint}</p><div className="placeholder-lines" aria-hidden="true"><span/><span/><span/></div></div>}
        <p className="result-note">ⓘ {t.note}</p>
      </aside>
    </div>
    <div className="steps">{t.steps.map(step => <span key={step}>{step}</span>)}</div>
  </>;
}
