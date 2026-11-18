// GET /api/status: is the model server on Render awake?
// A free Render service sleeps after 15 minutes. Any request wakes it, but it needs about a minute to start.
// We ask a small page of the model API: an answer = awake; no answer in time = still waking.
// The request itself is what wakes Render up, so the page simply calls this every few seconds.
// The timeout is long on purpose: right after Render starts, the FIRST call to this page took ~47 s
// (measured 18 Nov). With a short timeout we kept giving up and the page said "waking up" for minutes.

export async function onRequestGet({ env }) {
  if (env.MOCK_MODEL === "1") return Response.json({ awake: true, mock: true });
  return Response.json({ awake: await isAwake(env) });
}

export async function isAwake(env, ms = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    const response = await fetch(`${env.MODEL_URL}/gradio_api/info`, { signal: controller.signal });
    return response.ok;   // 502/503 while Render starts = not awake
  } catch {
    return false; // timeout or connection refused: Render is still starting
  } finally {
    clearTimeout(timer);
  }
}
