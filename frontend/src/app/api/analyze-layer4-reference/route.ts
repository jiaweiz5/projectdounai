export const runtime = "nodejs";

export const maxDuration = 120;

export async function POST(request: Request) {
  const base = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
  const token = process.env.BACKEND_SHARED_TOKEN;
  let localBackend = false;
  try {
    localBackend = ["127.0.0.1", "localhost", "[::1]"].includes(new URL(base).hostname);
  } catch {
    return Response.json({ error: "Invalid backend configuration" }, { status: 503 });
  }

  if (!localBackend && !token) {
    return Response.json(
      { error: "Server configuration missing" },
      { status: 503 }
    );
  }

  try {
    // Keep the files as multipart form data all the way to FastAPI.
    const formData = await request.formData();

    const upstream = await fetch(`${base}/analyze-layer4-reference`, {
      method: "POST",
      headers: token ? { "X-Backend-Token": token } : {},
      body: formData,
      cache: "no-store",
      signal: AbortSignal.timeout(105000),
    });

    if (!upstream.ok) {
      return Response.json(
        { error: "Layer 4 reference analysis unavailable" },
        { status: [413, 415, 422].includes(upstream.status) ? upstream.status : 502 }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch (error) {
    console.error("Layer 4 reference proxy error:", error);

    return Response.json(
      {
        error:
          "Please retry the Layer 4 reference analysis",
      },
      { status: 503 }
    );
  }
}
