// User accounts: password hashing and the signed session cookie.
// The shared team password (APP_PASSWORD) stays the first gate: it is asked at every sign-up and log-in.

const enc = new TextEncoder();
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
const unhex = (s) => new Uint8Array(s.match(/../g).map((h) => parseInt(h, 16)));
export const SESSION_DAYS = 7;

// Same length, compare every character (no early exit), so timing doesn't reveal how much matched
function sameText(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// ---- Passwords: never stored, only a slow salted hash (PBKDF2-SHA256, 100,000 rounds) ----
const ROUNDS = 100000;   // the most Cloudflare Workers allow for PBKDF2

async function pbkdf2(password, salt, rounds) {
  const key = await crypto.subtle.importKey("raw", enc.encode(password), "PBKDF2", false, ["deriveBits"]);
  return hex(await crypto.subtle.deriveBits({ name: "PBKDF2", salt, iterations: rounds, hash: "SHA-256" }, key, 256));
}

export async function hashPassword(password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  return `pbkdf2$${ROUNDS}$${hex(salt)}$${await pbkdf2(password, salt, ROUNDS)}`;
}

export async function checkPassword(password, stored) {
  const [kind, rounds, salt, hash] = (stored || "").split("$");
  if (kind !== "pbkdf2" || !hash) return false;
  return sameText(await pbkdf2(password, unhex(salt), Number(rounds)), hash);
}

// ---- The team password, compared as hashes ----
export async function teamPasswordOk(env, password) {
  if (!env.APP_PASSWORD) return false;
  const h = async (p) => hex(await crypto.subtle.digest("SHA-256", enc.encode(`hackathon:${p}`)));
  return sameText(await h(password || ""), await h(env.APP_PASSWORD));
}

// ---- Session cookie: "<user id>.<expiry>.<signature>" ----
// Signed with a key derived from APP_PASSWORD: it can't be forged or edited to another user's id,
// and changing the team password logs everyone out.
async function sign(env, message) {
  const key = await crypto.subtle.importKey("raw", enc.encode(`session:${env.APP_PASSWORD}`),
                                            { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return hex(await crypto.subtle.sign("HMAC", key, enc.encode(message)));
}

export async function sessionCookie(env, userId, secure) {
  const exp = Math.floor(Date.now() / 1000) + SESSION_DAYS * 86400;
  const value = `${userId}.${exp}.${await sign(env, `${userId}.${exp}`)}`;
  // HttpOnly: page scripts can't read it; SameSite=Strict: not sent by other sites
  return `session=${value}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${SESSION_DAYS * 86400}${secure ? "; Secure" : ""}`;
}

// Returns the user id, or null if the cookie is missing, edited or expired
export async function userFromCookie(env, cookieHeader) {
  const value = (cookieHeader || "").match(/(?:^|;\s*)session=([0-9a-f-]{36}\.\d{1,12}\.[0-9a-f]{64})/)?.[1];
  if (!value || !env.APP_PASSWORD) return null;
  const [userId, exp, sig] = value.split(".");
  if (Number(exp) < Date.now() / 1000) return null;
  return sameText(await sign(env, `${userId}.${exp}`), sig) ? userId : null;
}
