/* NEURAL / LAB core: the published model in the browser (ONNX Runtime Web), colour maps and the shared weld.
   Everything returned here is real model output. Nothing is generated or animated from made-up values. */
"use strict";
const NL = (() => {
  const N = 64, NN = N * N;
  const CLASS_COL = { no_defect: "#8fb996", porosity: "#e2b13c", crack: "#e0604f", lack_of_penetration: "#86a9e6" };
  const CLASS_NAME = { no_defect: "sound weld", porosity: "porosity", crack: "crack", lack_of_penetration: "lack of penetration" };
  const HEAT = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];       // = deployed app
  const SEQ = [[12, 12, 13], [74, 42, 28], [232, 116, 44], [246, 236, 214]];                    // activation
  let cfg = null, enc = null, dec = null, queue = Promise.resolve();
  const listeners = new Set();
  const state = { weld: null, orig: null, cur: null, base: null, now: null, z: null };

  const json = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };
  const serial = fn => { const r = queue.then(fn); queue = r.catch(() => {}); return r; };   // sessions are not re-entrant
  const f32 = t => Float32Array.from(t.data);

  async function init() {
    ort.env.wasm.numThreads = 1;
    cfg = await json("/api/config");
    [enc, dec] = await Promise.all([ort.InferenceSession.create("/models/encoder.onnx"),
                                    ort.InferenceSession.create("/models/decoder.onnx")]);
    return cfg;
  }
  const decodeNow = async z => {
    const o = await dec.run({ latent: new ort.Tensor("float32", z, [1, 32]) });
    return { dec1: f32(o.dec1), dec2: f32(o.dec2), dec3: f32(o.dec3), rec: f32(o.reconstruction) };
  };
  /* full forward pass of one 64x64 tile: encoder taps, latent, decoder taps, reconstruction, error, score */
  const run = x => serial(async () => {
    const e = await enc.run({ image: new ort.Tensor("float32", x, [1, 1, N, N]) });
    const z = f32(e.latent), d = await decodeNow(z);
    return { x, z, enc1: f32(e.block1), enc2: f32(e.block2), enc3: f32(e.block3), ...d, ...errorOf(x, d.rec) };
  });
  const decode = z => serial(() => decodeNow(z));
  function errorOf(x, rec) {
    const err = new Float32Array(NN); let s = 0;
    for (let i = 0; i < NN; i++) { err[i] = (x[i] - rec[i]) ** 2; s += err[i]; }
    return { err, score: s / NN };
  }
  const level = s => s > cfg.t_high ? "high" : s > cfg.t_low ? "check" : "low";
  const LEVEL_TEXT = { low: "Low", check: "Check", high: "High" };
  const LEVEL_SENTENCE = { low: "Looks like a sound weld.", check: "Unusual. An inspector should look.", high: "Strongly abnormal. Inspect first." };

  /* colour */
  const lerpStops = (S, v) => { const p = Math.min(Math.max(v, 0), 1) * (S.length - 1), lo = Math.min(Math.floor(p), S.length - 2), f = p - lo;
    return [0, 1, 2].map(c => S[lo][c] * (1 - f) + S[lo + 1][c] * f); };
  const divRGB = v => { const t = Math.min(Math.abs(v), 1), b = [12, 12, 13], g = v < 0 ? [110, 160, 235] : [232, 116, 44];
    return b.map((x, i) => x + (g[i] - x) * t); };
  /* paint an H x H slice of a flat array (offset = channel * H * H) */
  function paint(canvas, arr, H, kind = "grey", scale = 1, offset = 0) {
    if (canvas.width !== H) { canvas.width = H; canvas.height = H; }
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(H, H), d = img.data;
    for (let i = 0; i < H * H; i++) {
      const v = arr[offset + i]; let r, g, b;
      if (kind === "heat") [r, g, b] = lerpStops(HEAT, v / scale);
      else if (kind === "act") [r, g, b] = lerpStops(SEQ, v / scale);
      else if (kind === "div") [r, g, b] = divRGB(v / scale);
      else { r = g = b = Math.min(Math.max(v, 0), 1) * 255; }
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  /* per-channel statistics of a (C, H, H) tensor */
  function channelStats(arr, C, H) {
    const HH = H * H, mean = new Float32Array(C); let max = 0, zeros = 0;
    for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < HH; i++) { const v = arr[c * HH + i]; s += v; if (v > max) max = v; if (v <= 0) zeros++; } mean[c] = s / HH; }
    const order = [...mean.keys()].sort((a, b) => mean[b] - mean[a]);
    return { mean, order, max, zeroShare: zeros / (C * HH), overall: mean.reduce((a, b) => a + b, 0) / C };
  }

  /* shared weld: the tile every room works on */
  async function setTile(t) {
    state.weld = t; state.orig = Float32Array.from(t.x); state.cur = state.orig.slice();
    state.base = await run(state.orig.slice()); state.now = state.base; state.z = state.base.z.slice();
    state.match = Math.abs(state.base.score - t.python_score) <= 1e-7 + 1e-3 * t.python_score;
    listeners.forEach(f => f("tile"));
  }
  const loadTest = async id => setTile(await json(`/api/tile/${id}`));
  async function loadFile(file) {
    const r = await fetch("/api/preprocess", { method: "POST", body: file }), t = await r.json();
    if (t.error) throw new Error(t.error);
    return setTile(t);
  }
  const on = f => listeners.add(f);
  const emit = what => listeners.forEach(f => f(what));

  return { N, NN, CLASS_COL, CLASS_NAME, LEVEL_TEXT, LEVEL_SENTENCE, init, run, decode, errorOf, level, paint, channelStats,
           state, loadTest, loadFile, on, emit, json, get cfg() { return cfg; } };
})();
