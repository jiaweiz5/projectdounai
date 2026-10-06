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
    const body = await request.json();

    const upstream = await fetch(`${base}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Backend-Token": token,
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(105000),
    });

    if (!upstream.ok) {
      const upstreamError = await upstream.text();

      console.error(
        "Backend analysis failed:",
        upstream.status,
        upstreamError
      );

      return Response.json(
        { error: "Analysis unavailable" },
        { status: 502 }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch (error) {
    console.error("Analyze proxy error:", error);

    return Response.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Please retry the analysis",
      },
      { status: 503 }
    );
  }
}
