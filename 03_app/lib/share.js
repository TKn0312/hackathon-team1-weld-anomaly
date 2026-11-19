// Signed links: let NEURAL / LAB · Look inside (on Render) open the SAME X-ray the inspector checked here.
// The lab has no password, so it gets a link to ONE image that works without the login cookie:
//   /api/shared/<id>?exp=<unix seconds>&sig=<HMAC-SHA256 of "<id>.<exp>">
// The signature is made with a key derived from APP_PASSWORD, so only this app can make links;
// changing <id> or <exp> breaks the signature, and the link stops working after LINK_HOURS.
const LINK_HOURS = 24;

async function sign(env, message) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(`share:${env.APP_PASSWORD}`),
                                            { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return [...new Uint8Array(mac)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// The address of the lab (MODEL_URL without /gradio), opened at the Architecture room with this weld
export async function labUrl(env, origin, id) {
  const exp = Math.floor(Date.now() / 1000) + LINK_HOURS * 3600;
  const image = `${origin}/api/shared/${id}?exp=${exp}&sig=${await sign(env, `${id}.${exp}`)}`;
  const lab = env.MODEL_URL.replace(/\/gradio\/?$/, "");
  return `${lab}/?weld=${encodeURIComponent(image)}#architecture`;
}

export async function validLink(env, id, exp, sig) {
  if (!/^\d{1,12}$/.test(exp || "") || !/^[a-f0-9]{64}$/.test(sig || "")) return false;
  if (Number(exp) < Date.now() / 1000) return false;               // expired
  const expected = await sign(env, `${id}.${exp}`);
  let diff = 0;                                                     // compare every character (no early exit)
  for (let i = 0; i < 64; i++) diff |= expected.charCodeAt(i) ^ sig.charCodeAt(i);
  return diff === 0;
}
