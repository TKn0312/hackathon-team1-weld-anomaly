/* NEURAL / LAB · Latent universe room.
   Each point is one of the 3,019 test tiles, placed by a PCA projection of its real 32-number code and coloured by
   its TRUE dataset class (the model never saw these labels). Your current weld is projected with the same PCA. */
"use strict";
const Universe = (() => {
  let root, cv, ctx, pts = [], mode = "3d", yaw = -0.6, pitch = 0.35, drag = null, hover = -1, raf = 0, hidden = new Set();
  let thumbs = new Map(), lastInteract = 0, box = null;
  const $ = s => root.querySelector(s);

  async function build(el) {
    root = el; cv = $("#uCanvas"); ctx = cv.getContext("2d");
    pts = await NL.json("/api/map");
    const ext = k => { const v = pts.map(p => p[k]); return [Math.min(...v), Math.max(...v)]; };
    box = { x: ext("x"), y: ext("y"), z: ext("z") };
    const ex = NL.cfg.pca_explained.map(v => Math.round(100 * v));
    $("#uAxes").textContent = `Principal component analysis of the 32-number codes. PC1 ${ex[0]}%, PC2 ${ex[1]}%, PC3 ${ex[2]}% of their variance. A projection: distances are approximate.`;
    const counts = {}; pts.forEach(p => counts[p.label] = (counts[p.label] || 0) + 1);
    $("#uKey").innerHTML = Object.keys(NL.CLASS_COL).map(c => `<button class="ukey on" data-c="${c}"><i style="background:${NL.CLASS_COL[c]}"></i>${NL.CLASS_NAME[c]} <span class="num">${counts[c]}</span></button>`).join("");
    root.querySelectorAll(".ukey").forEach(b => b.onclick = () => { b.classList.toggle("on"); b.classList.contains("on") ? hidden.delete(b.dataset.c) : hidden.add(b.dataset.c); draw(); });
    root.querySelectorAll("[data-umode]").forEach(b => b.onclick = () => { root.querySelectorAll("[data-umode]").forEach(o => o.classList.remove("on")); b.classList.add("on"); mode = b.dataset.umode; draw(); });
    cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, yaw, pitch, moved: false }; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", e => {
      lastInteract = performance.now();
      if (drag && mode === "3d") { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        yaw = drag.yaw + dx * .008; pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + dy * .008)); draw(); return; }
      pick(e); });
    cv.addEventListener("pointerup", e => { const was = drag; drag = null; if (was && !was.moved && hover >= 0) select(hover); });
    cv.addEventListener("pointerleave", () => { hover = -1; $("#uTip").hidden = true; draw(); });
    addEventListener("resize", () => { if (!root.hidden) { fit(); draw(); } });
    NL.on(() => { if (!root.hidden) draw(); });
  }
  function fit() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; }

  /* normalised coordinates -> screen */
  const norm = (v, [a, b]) => (v - (a + b) / 2) / ((b - a) / 2);
  function project(p) {
    const W = cv.width, H = cv.height, s = Math.min(W, H) * .42;
    let x = norm(p.x, box.x), y = norm(p.y, box.y), z = norm(p.z, box.z);
    if (mode === "2d") return [W / 2 + x * W * .44, H / 2 - y * H * .42, 0];
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    const persp = 1 / (1 + z2 * .18);
    return [W / 2 + x1 * s * persp * 1.25, H / 2 - y1 * s * persp, z2];
  }
  function current() {   // the shared weld (as drawn) projected with the same PCA
    if (!NL.state.now) return null;
    const m = NL.cfg.pca_mean, P = NL.cfg.pca_components, z = NL.state.now.z, c = [0, 0, 0];
    for (let k = 0; k < 3; k++) for (let d = 0; d < 32; d++) c[k] += (z[d] - m[d]) * P[k][d];
    return { x: c[0], y: c[1], z: c[2] };
  }
  function draw() {
    if (!pts.length || root.hidden) return;
    const W = cv.width, H = cv.height, d = devicePixelRatio || 1;
    ctx.clearRect(0, 0, W, H);
    const pr = pts.map((p, i) => [i, ...project(p)]).filter(([i]) => !hidden.has(pts[i].label));
    if (mode === "3d") pr.sort((a, b) => b[3] - a[3]);
    for (const [i, x, y, z] of pr) {
      const r = (mode === "3d" ? 2.2 - z * .7 : 2) * d;
      ctx.globalAlpha = mode === "3d" ? .45 + .35 * (1 - (z + 1) / 2) : .6;
      ctx.fillStyle = NL.CLASS_COL[pts[i].label]; ctx.beginPath(); ctx.arc(x, y, Math.max(r, 1), 0, 6.3); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (hover >= 0) { const [x, y] = project(pts[hover]); ring(x, y, 9 * d, "#efe8da", 1.5 * d); }
    const c = current();
    if (c) { const [x, y] = project(c); ring(x, y, 13 * d, "#e8742c", 2.5 * d); ring(x, y, 4 * d, "#e8742c", 4 * d);
      ctx.fillStyle = "#efe8da"; ctx.font = `${13 * d}px "IBM Plex Sans", sans-serif`; ctx.fillText("your weld", x + 18 * d, y + 4 * d); }
  }
  function ring(x, y, r, col, w) { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.3); ctx.stroke(); }

  /* hover: nearest visible point; show the real tile and its real score */
  let tipTimer = 0;
  function pick(e) {
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1, mx = (e.clientX - r.left) * d, my = (e.clientY - r.top) * d;
    let best = -1, bd = (14 * d) ** 2;
    pts.forEach((p, i) => { if (hidden.has(p.label)) return; const [x, y] = project(p), dd = (x - mx) ** 2 + (y - my) ** 2; if (dd < bd) { bd = dd; best = i; } });
    if (best !== hover) { hover = best; draw(); }
    const tip = $("#uTip");
    if (best < 0) { tip.hidden = true; return; }
    const p = pts[best];
    tip.hidden = false; tip.style.left = (e.clientX - r.left + 18) + "px"; tip.style.top = (e.clientY - r.top + 18) + "px";
    $("#uTipText").innerHTML = `<b>${NL.CLASS_NAME[p.label]}</b><br>test weld ${p.weld}<br><span class="num">score ${p.score.toFixed(6)}</span> · ${NL.LEVEL_TEXT[NL.level(p.score)]}<br><span class="dim">click to load</span>`;
    clearTimeout(tipTimer);
    tipTimer = setTimeout(async () => { if (!thumbs.has(best)) thumbs.set(best, (await NL.json(`/api/tile/${best}`)).x);
      if (hover === best) NL.paint($("#uTipImg"), thumbs.get(best), 64); }, 90);
    if (thumbs.has(best)) NL.paint($("#uTipImg"), thumbs.get(best), 64);
  }
  async function select(i) { await NL.loadTest(i); $("#uLoaded").textContent = `Loaded: ${NL.CLASS_NAME[pts[i].label]}, test weld ${pts[i].weld}. Open the Defect experiment or the Architecture to work on it.`; }

  /* gentle turn of the 3-D view while nobody touches it (a camera move only; the data do not change) */
  function loop() {
    raf = requestAnimationFrame(loop);
    if (root.hidden || mode !== "3d" || drag || performance.now() - lastInteract < 2500 || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    yaw += .0018; draw();
  }
  function show() { fit(); draw(); if (!raf) loop(); }
  return { build, show, draw, select, get points() { return pts; } };
})();
