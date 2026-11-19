// POST /api/login, a form with: team_password, username, password, and mode = "login" or "signup".
// The shared team password is always required first; then each person logs in with their own account.
import { teamPasswordOk, hashPassword, checkPassword, sessionCookie } from "../../lib/users.js";

const USERNAME = /^[a-z0-9_.-]{3,32}$/;

export async function onRequestPost({ request, env }) {
  const form = await request.formData();
  const back = (error) => Response.redirect(new URL(`/login?error=${error}`, request.url), 302);

  if (!(await teamPasswordOk(env, form.get("team_password")))) return back("team");

  const username = String(form.get("username") || "").trim().toLowerCase();
  const password = String(form.get("password") || "");
  if (!USERNAME.test(username) || password.length < 8 || password.length > 200) return back("invalid");

  let userId;
  if (form.get("mode") === "signup") {
    userId = crypto.randomUUID();
    try {
      // "?" placeholders: the username never goes into the SQL text
      await env.DB.prepare("INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)")
        .bind(userId, username, await hashPassword(password), new Date().toISOString()).run();
    } catch {
      return back("taken");        // UNIQUE(username): someone already has this name
    }
  } else {
    const user = await env.DB.prepare("SELECT id, password_hash FROM users WHERE username = ?").bind(username).first();
    // Same answer for "no such user" and "wrong password": don't reveal which usernames exist
    if (!user || !(await checkPassword(password, user.password_hash))) return back("wrong");
    userId = user.id;
  }

  const secure = new URL(request.url).protocol === "https:";
  return new Response(null, {
    status: 302,
    headers: { Location: "/", "Set-Cookie": await sessionCookie(env, userId, secure) },
  });
}
