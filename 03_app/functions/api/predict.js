// POST /api/predict with a form field "image" (a PNG or JPEG file):
// 1. send it to the model server, 2. store the image AND the error map in R2, 3. store the result in D1,
// 4. answer the page.
// If the model server on Render is asleep or starting (no answer in time, or 502/503/504), we answer
// { waking: true } with status 503: the page then shows "Model is waking up, about 1 minute…" and retries.
import { isAwake } from "./status.js";
import { labUrl } from "../../lib/share.js";

export async function onRequestPost({ request, env, data }) {
  const form = await request.formData();
  const file = form.get("image");
  if (!file || typeof file === "string") {
    return Response.json({ error: "Send an image file in the field 'image'." }, { status: 400 });
  }
  if (file.size > 5 * 1024 * 1024) {
    return Response.json({ error: "Image too large (max 5 MB)." }, { status: 413 });
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  // Security: trust the file's first bytes, not its name or the browser's type. Only real PNG/JPEG get through,
  // so nothing else (e.g. an HTML page) can be stored in R2 and served back from our address.
  const contentType = imageType(bytes);
  if (!contentType) {
    return Response.json({ error: "Please upload a PNG or JPEG image." }, { status: 415 });
  }

  // 1. Ask the model
  let result;
  if (env.MOCK_MODEL === "1") {
    result = mockModel(toBase64(bytes));
  } else {
    // Quick check first: if Render is asleep, don't hold this request for a whole minute
    if (!(await isAwake(env, 25000))) return waking();
    try {
      result = await callModel(env, toBase64(bytes));
    } catch (err) {
      console.error("model call failed:", err);   // details go to the Cloudflare logs, not to the browser
      if (err.waking) return waking();
      return Response.json({ error: "The model server could not analyse this image." }, { status: 502 });
    }
  }

  // 2. Store the image and the error map in R2, under a random id
  const id = crypto.randomUUID();
  const imageKey = `uploads/${id}`;
  await env.BUCKET.put(imageKey, bytes, { httpMetadata: { contentType } });
  const errorMap = fromBase64(result.error_map_png_base64 || "");
  if (imageType(errorMap) === "image/png") {
    await env.BUCKET.put(`errors/${id}`, errorMap, { httpMetadata: { contentType: "image/png" } });
  }

  // 3. Store one row in D1. The "?" placeholders keep user input out of the SQL text
  const createdAt = new Date().toISOString();
  await env.DB.prepare(
    "INSERT INTO predictions (id, created_at, image_key, file_name, score, threshold, is_anomaly, risk_level, score_vs_threshold, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(id, createdAt, imageKey, file.name || null, result.score, result.threshold, result.is_anomaly ? 1 : 0,
          result.risk_level, result.score_vs_threshold, data.userId)   // whose check this is (from the session cookie)
    .run();

  // 4. Answer the page, with a signed link that opens this X-ray in NEURAL / LAB · Look inside
  const lab_url = await labUrl(env, new URL(request.url).origin, id);
  return Response.json({ id, created_at: createdAt, lab_url, ...result });
}

const waking = () => Response.json({ waking: true }, { status: 503 });

// MOCK: a made-up answer with the SAME fields as our model server (02_model_server/README.md).
// The thresholds are the real ones; the score is random; the two images are just the input sent back.
function mockModel(imageBase64) {
  const threshold = 0.000345, t_low = 0.000255, t_high = 0.001442;
  const score = 0.00012 * Math.pow(30, Math.random()); // random between 0.00012 and 0.0036
  const risk_level = score <= t_low ? "low" : score > t_high ? "high" : "check";
  return {
    score,
    risk_level,
    is_anomaly: score > threshold,
    score_vs_threshold: Math.round((score / threshold) * 100) / 100,
    threshold,
    t_low,
    t_high,
    reconstruction_png_base64: imageBase64, // placeholder: the real server sends the rebuilt tile
    error_map_png_base64: imageBase64,      // placeholder: the real server sends the heatmap
    mock: true,
  };
}

// Gradio's HTTP API has two steps: start a job, then read its result.
// A timeout or a 502/503/504 from Render means "still starting": we mark the error as waking.
async function callModel(env, imageBase64) {
  const base = `${env.MODEL_URL}/gradio_api/call/${env.MODEL_API_NAME}`;
  const started = await fetchModel(base, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: [imageBase64] }),
  });
  const { event_id } = await started.json();

  const stream = await fetchModel(`${base}/${event_id}`);
  const text = await stream.text(); // "event: complete\ndata: [ {...} ]"
  const complete = text.split("event: complete")[1];
  if (!complete) throw new Error(`No result from the model server: ${text.slice(0, 200)}`);
  const data = JSON.parse(complete.split("data:")[1].trim());
  return data[0];
}

async function fetchModel(url, options = {}, ms = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  let response;
  try {
    response = await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    throw Object.assign(new Error(`No answer from the model server: ${err}`), { waking: true });
  } finally {
    clearTimeout(timer);
  }
  if ([502, 503, 504].includes(response.status)) {
    throw Object.assign(new Error(`Model server answered ${response.status}`), { waking: true });
  }
  if (!response.ok) throw new Error(`Model server answered ${response.status}`);
  return response;
}

// The "magic numbers" at the start of the file: PNG = 89 50 4E 47, JPEG = FF D8 FF
function imageType(b) {
  if (b.length > 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return "image/png";
  if (b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "image/jpeg";
  return null;
}

function toBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function fromBase64(text) {
  try {
    const binary = atob(text);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  } catch {
    return new Uint8Array(0);
  }
}
