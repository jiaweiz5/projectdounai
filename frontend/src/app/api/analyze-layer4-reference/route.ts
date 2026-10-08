export const runtime = "nodejs";

export const maxDuration = 120;

export async function POST(request: Request) {
  const base = process.env.BACKEND_URL;
  const token = process.env.BACKEND_SHARED_TOKEN;

  if (!base || !token) {
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
      headers: {
        "X-Backend-Token": token,
      },
      body: formData,
      cache: "no-store",
      signal: AbortSignal.timeout(105000),
    });

    if (!upstream.ok) {
      const upstreamError = await upstream.text();

      console.error(
        "Backend Layer 4 reference analysis failed:",
        upstream.status,
        upstreamError
      );

      return Response.json(
        { error: "Layer 4 reference analysis unavailable" },
        { status: 502 }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch (error) {
    console.error("Layer 4 reference proxy error:", error);

    return Response.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Please retry the Layer 4 reference analysis",
      },
      { status: 503 }
    );
  }
}
