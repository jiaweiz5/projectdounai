"use client";

import { useState } from "react";
import type { ChangeEvent } from "react";
import Layer3ScreenshotAnalyzer from "@/components/Layer3ScreenshotAnalyzer";

type Comment = {
  id: string;
  text: string;
  timestamp: null;
};

export default function Home() {
  const [text, setText] = useState("");
  const [comments, setComments] = useState<Comment[]>([]);
  const [commentInput, setCommentInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<unknown>(null);

  function addComment() {
    const trimmed = commentInput.trim();

    if (!trimmed) return;

    if (comments.length >= 30) {
      setError("Maximum of 30 comments allowed.");
      return;
    }

    const newComment: Comment = {
      id: `c${String(comments.length + 1).padStart(2, "0")}`,
      text: trimmed,
      timestamp: null,
    };

    setComments((previous) => [...previous, newComment]);
    setCommentInput("");
    setError("");
  }

  function removeComment(id: string) {
    setComments((previous) =>
      previous.filter((comment) => comment.id !== id),
    );
  }

  function handleImages(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);

    if (files.length > 2) {
      setError("You can upload a maximum of 2 images.");
      event.target.value = "";
      return;
    }

    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);

    if (totalBytes > 2 * 1024 * 1024) {
      setError("Images must be under 2 MB total.");
      event.target.value = "";
      return;
    }

    setError("");

    Promise.all(
      files.map(
        (file) =>
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(new Error("Could not read image"));
            reader.readAsDataURL(file);
          }),
      ),
    )
      .then((dataUrls) => {
        setImages(dataUrls);
      })
      .catch(() => {
        setError("Could not read the selected image.");
      });
  }

  async function analyze() {
    if (loading) return;

    if (!text.trim()) {
      setError("Please enter some post text.");
      return;
    }

    const input = {
  text,
  comments: comments.map((comment) => comment.text),
  images,
};

    const serialized = JSON.stringify(input);
    const requestBytes = new Blob([serialized]).size;

    if (requestBytes > 3.5 * 1024 * 1024) {
      setError("Request is too large.");
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

      if (!response.ok) {
        setError("Analysis unavailable");
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setError("Analysis unavailable");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-100 px-6 py-10">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Title */}
        <div>
          <h1 className="text-3xl font-bold text-zinc-900">
            XHS Content Verifier
          </h1>

          <p className="mt-2 text-zinc-600">
            Analyze Xiaohongshu posts, comments, and images for authenticity.
          </p>
        </div>

        {/* Layer 3 screenshot upload and coordination detection */}
        <Layer3ScreenshotAnalyzer />

        {/* Original text, image, and comment analyzer */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            Post Text
          </h2>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={10000}
            placeholder="Paste the Xiaohongshu post text here..."
            className="min-h-48 w-full rounded-lg border border-zinc-300 p-4 text-zinc-900 outline-none focus:border-zinc-500"
          />

          <div className="mt-2 text-right text-sm text-zinc-500">
            {text.length} / 10,000
          </div>
        </section>

        {/* Images */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">Images</h2>

          <p className="mb-3 text-sm text-zinc-500">
            Optional. Maximum 2 images and 2 MB total.
          </p>

          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleImages}
            className="text-zinc-700"
          />

          {images.length > 0 && (
            <p className="mt-3 text-sm text-zinc-600">
              {images.length} image{images.length !== 1 ? "s" : ""} selected
            </p>
          )}
        </section>

        {/* Comments */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            Comments
          </h2>

          <div className="flex gap-2">
            <input
              type="text"
              value={commentInput}
              onChange={(event) => setCommentInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addComment();
              }}
              placeholder="Add a comment..."
              className="flex-1 rounded-lg border border-zinc-300 px-4 py-2 text-zinc-900 outline-none focus:border-zinc-500"
            />

            <button
              type="button"
              onClick={addComment}
              disabled={comments.length >= 30}
              className="rounded-lg bg-zinc-800 px-5 py-2 text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Add
            </button>
          </div>

          <p className="mt-2 text-sm text-zinc-500">
            {comments.length} / 30 comments
          </p>

          <div className="mt-4 space-y-2">
            {comments.map((comment) => (
              <div
                key={comment.id}
                className="flex items-center justify-between rounded-lg bg-zinc-100 p-3"
              >
                <div>
                  <span className="mr-2 text-xs font-semibold text-zinc-500">
                    {comment.id}
                  </span>

                  <span className="text-zinc-800">{comment.text}</span>
                </div>

                <button
                  type="button"
                  onClick={() => removeComment(comment.id)}
                  className="ml-4 text-sm text-red-600 hover:text-red-800"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Error from the original analyzer */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}

        {/* Analyze button for the original analyzer */}
        <button
          type="button"
          onClick={analyze}
          disabled={loading}
          className="w-full rounded-xl bg-black px-6 py-4 font-semibold text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {/* JSON result from the original analyzer */}
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-lg font-semibold text-zinc-900">
            Analysis Result
          </h2>

          {result ? (
            <pre className="overflow-x-auto rounded-lg bg-zinc-950 p-4 text-sm text-green-400">
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : (
            <div className="rounded-lg bg-zinc-100 p-4 text-zinc-500">
              Analysis results will appear here.
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
