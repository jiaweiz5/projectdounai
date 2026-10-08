/**
 * Next.js route for POST /api/analyze.
 * Place this file at frontend/src/app/api/analyze/route.ts.
 * It forwards the main form's JSON to the FastAPI /analyze endpoint.
 */

// Use a configurable backend address, defaulting to the local port in use.
const backendUrl = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

export async function POST(request: Request): Promise<Response> {
  // Preserve the JSON body exactly as the frontend sent it to this route.
  const body = await request.text();

  try {
    // Forward the request to FastAPI rather than handling the model in Next.js.
    const backendResponse = await fetch(`${backendUrl}/analyze`, {
      method: "POST",
      headers: {
        "content-type": request.headers.get("content-type") ?? "application/json",
      },
      body,
      cache: "no-store",
    });

    // Return FastAPI's status and body, including helpful validation errors.
    return new Response(await backendResponse.text(), {
      status: backendResponse.status,
      headers: {
        "content-type":
          backendResponse.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    // A network failure means Next.js could not reach the running backend.
    return Response.json(
      { detail: "The analysis backend is unavailable." },
      { status: 502 },
    );
  }
}
