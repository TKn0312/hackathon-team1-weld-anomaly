// GET /api/history: the logged-in user's 20 most recent predictions, read from D1 (never other users' rows),
// each with a fresh signed link that opens its X-ray in NEURAL / LAB · Look inside
import { labUrl } from "../../lib/share.js";

export async function onRequestGet({ request, env, data }) {
  const { results } = await env.DB.prepare(
    "SELECT id, created_at, file_name, score, threshold, is_anomaly, risk_level, score_vs_threshold FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
  ).bind(data.userId).all();
  const origin = new URL(request.url).origin;
  for (const row of results) row.lab_url = await labUrl(env, origin, row.id);
  return Response.json(results);
}
