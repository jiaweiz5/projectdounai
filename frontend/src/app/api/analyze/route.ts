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
      return Response.json(
        { error: "Analysis unavailable" },
        { status: 502 }
      );
    }

    return Response.json(await upstream.json());
  } catch {
    return Response.json(
      { error: "Please retry the analysis" },
      { status: 503 }
    );
  }
}