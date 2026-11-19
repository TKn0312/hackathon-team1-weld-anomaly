// GET /api/images/<id>            the uploaded X-ray, read back from R2
// GET /api/images/<id>?kind=error the error heat map the model returned for it
// (behind the password like everything else)
const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;   // the ids we create (crypto.randomUUID)
const IMAGE_TYPES = ["image/png", "image/jpeg"];

export async function onRequestGet({ params, request, env, data }) {
  if (!ID.test(params.id)) {
    return new Response("Not found", { status: 404 });
  }
  // Own data only: the image must belong to a check of the logged-in user (same answer as "does not exist")
  const own = await env.DB.prepare("SELECT 1 FROM predictions WHERE id = ? AND user_id = ?").bind(params.id, data.userId).first();
  if (!own) {
    return new Response("Not found", { status: 404 });
  }
  const kind = new URL(request.url).searchParams.get("kind");
  const key = kind === "error" ? `errors/${params.id}` : `uploads/${params.id}`;
  const object = await env.BUCKET.get(key);
  if (!object) {
    return new Response("Not found", { status: 404 });
  }
  const type = object.httpMetadata?.contentType;
  return new Response(object.body, {
    headers: {
      // Security: only ever serve as an image, and tell the browser not to guess another type
      "Content-Type": IMAGE_TYPES.includes(type) ? type : "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
      "Content-Security-Policy": "default-src 'none'",
      "Cache-Control": "private, max-age=3600",
    },
  });
}
