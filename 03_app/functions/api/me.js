// GET /api/me: who is logged in (for the "Logged in as …" line on the page)
export async function onRequestGet({ env, data }) {
  const user = await env.DB.prepare("SELECT username FROM users WHERE id = ?").bind(data.userId).first();
  if (!user) return Response.json({ error: "Not logged in" }, { status: 401 });
  return Response.json({ username: user.username });
}
