// Run this handler in Node because it forwards a multipart file upload.
export const runtime = "nodejs";

// Give the backend time to run OCR on the uploaded comment screenshot.
export const maxDuration = 120;

export async function POST(request: Request) {
  // These values stay on the server and are never sent to the browser.
  const base = process.env.BACKEND_URL;
  const token = process.env.BACKEND_SHARED_TOKEN;

  if (!base || !token) {
    return Response.json(
      { error: "Server configuration missing" },
      { status: 503 },
    );
  }

  try {
    // Read the uploaded screenshot as multipart form data.
    const formData = await request.formData();
    if (!formData.get("file")) {
      return Response.json(
        { error: "Choose a comment screenshot first." },
        { status: 400 },
      );
    }

    // Do not set Content-Type here: fetch must add the multipart boundary.
    const upstream = await fetch(`${base.replace(/\/$/, "")}/analyze-screenshot`, {
      method: "POST",
      headers: { "X-Backend-Token": token },
      body: formData,
      cache: "no-store",
      signal: AbortSignal.timeout(105000),
    });

    // Preserve the backend status and JSON response for the frontend handler.
    const body = await upstream.text();
    if (!upstream.ok) {
      console.error(
        "Screenshot analysis failed:",
        upstream.status,
        body.slice(0, 1000),
      );
    }

    return new Response(body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    // A connection failure or timeout becomes a readable proxy response.
    console.error("Screenshot proxy error:", error);
    return Response.json(
      { error: "Screenshot analysis is unavailable. Please retry." },
      { status: 503 },
    );
  }
}
