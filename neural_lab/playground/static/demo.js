/* NEURAL / LAB · 90-second guided demo for the classroom.
   Every step drives the real model (real tiles, real inference). Captions that describe a result are written
   from the measured result, so the demo never claims an outcome the model did not produce. */
"use strict";
const Demo = (() => {
  let running = false, stopFlag = false, t0 = 0, timer = 0;
  const TOTAL = 90;
  const $ = id => document.getElementById(id);
  const sleep = ms => new Promise((res, rej) => { const t = setTimeout(res, ms); const c = setInterval(() => { if (stopFlag) { clearTimeout(t); clearInterval(c); rej("stop"); } }, 100); setTimeout(() => clearInterval(c), ms + 50); });
  const say = (title, text) => { $("demoTitle").textContent = title; $("demoText").textContent = text; };

  async function start(go) {
    if (running) return;
    running = true; stopFlag = false; t0 = performance.now(); document.body.classList.add("demo");
    $("demo").hidden = false;
    timer = setInterval(() => { $("demoBar").style.width = Math.min(100, (performance.now() - t0) / (TOTAL * 10)) + "%"; }, 200);
    const cat = NL.cfg.catalogue, sound = cat.find(t => t.label === "no_defect" && t.pick === "median");
    try {
      go("universe");
      say("The latent universe", "Every dot is one of 3,019 real X-ray tiles from welds the network never trained on, placed by the 32 numbers it compresses each tile into. Colours are the true classes, which the network never saw.");
      await sleep(14000);

      await NL.loadTest(sound.id); go("defect");
      say("A sound weld", `The network redraws it almost perfectly. Score ${NL.state.base.score.toFixed(6)}: ${NL.LEVEL_TEXT[NL.level(NL.state.base.score)]}.`);
      await sleep(10000);

      say("We paint a pore", "A synthetic dark spot, growing. The network has only ever learned sound welds. Watch the score.");
      const orig = NL.state.orig;
      for (let r = .5; r <= 5.01; r += .25) { NL.state.cur = Sens.withDefect(orig, "pore", r, .75); await Defect.apply(NL.state.cur); await sleep(160); }
      await sleep(600);                                    // let the last inference finish
      const s0 = NL.state.base.score, s1 = NL.state.now.score, L = NL.level(s1);
      say(`Score ×${(s1 / s0).toFixed(1)}`, L === "low"
        ? `The score rose from ${s0.toFixed(6)} to ${s1.toFixed(6)}, but stayed LOW: this time the network barely noticed.`
        : `The score rose from ${s0.toFixed(6)} to ${s1.toFixed(6)}: ${NL.LEVEL_TEXT[L].toUpperCase()}. A drawn spot is not a real defect, but the network reacts to anything it cannot redraw.`);
      await sleep(10000);

      say("Original, redraw, error", "Left: what we gave the network. Middle: what it could redraw. Right: the difference, at the deployed app's fixed scale.");
      document.querySelector(".seq").scrollIntoView({ block: "center", behavior: "smooth" });
      await Defect.playSequence(); await sleep(8000);

      go("architecture");
      say("Inside the network", "The painted weld passes through three encoder blocks. Each plane is a real feature map from this exact pass. The middle is the 32-number code: all the decoder receives.");
      await sleep(12000);

      const d = [...Array(32).keys()].sort((a, b) => NL.cfg.influence[b] - NL.cfg.influence[a])[0];
      const from = Arch.zMod[d], to = NL.cfg.latent_hi[d];
      say(`Steering number z${d}`, `We move one number, the one that measurably changes the redraw most, across its real range. The decoder answers instantly. We do not claim to know what it means.`);
      for (let k = 0; k <= 30; k++) { await Arch.setLatent(d, from + (to - from) * k / 30); await sleep(110); }
      await sleep(5000);

      go("universe");
      say("Where is our painted weld now?", "The orange ring is the painted weld, projected with the same PCA as every dot.");
      await sleep(12000);

      say("Everything you saw was computed live", "By Team 1's published model, in this browser. No pre-recorded frames. Your turn: draw, steer, explore.");
      await sleep(7000);
    } catch (e) { if (e !== "stop") console.error(e); }
    end();
  }
  function end() { running = false; clearInterval(timer); document.body.classList.remove("demo"); $("demo").hidden = true; }
  function stop() { stopFlag = true; }
  addEventListener("keydown", e => { if (e.key === "Escape" && running) stop(); });
  return { start, stop, get running() { return running; } };
})();
