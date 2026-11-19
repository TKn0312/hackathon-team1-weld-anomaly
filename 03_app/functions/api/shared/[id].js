// GET /api/shared/<id>?exp=...&sig=...  the uploaded X-ray from R2, WITHOUT the login cookie, for NEURAL / LAB.
// Only with a valid, unexpired signature made by this app (see lib/share.js); anything else gets 404.
import { validLink } from "../../../lib/share.js";

const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const IMAGE_TYPES = ["image/png", "image/jpeg"];

export async function onRequestGet({ params, request, env }) {
  const url = new URL(request.url);
  if (!ID.test(params.id) || !(await validLink(env, params.id, url.searchParams.get("exp"), url.searchParams.get("sig")))) {
    return new Response("Not found", { status: 404 });
  }
  const object = await env.BUCKET.get(`uploads/${params.id}`);
  if (!object) return new Response("Not found", { status: 404 });
  const type = object.httpMetadata?.contentType;
  return new Response(object.body, {
    headers: {
      "Content-Type": IMAGE_TYPES.includes(type) ? type : "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
      "Content-Security-Policy": "default-src 'none'",
      "Cache-Control": "private, max-age=3600",
      // The lab on Render reads it with fetch(): allow that other address to read the answer
      "Access-Control-Allow-Origin": "*",
    },
  });
}
