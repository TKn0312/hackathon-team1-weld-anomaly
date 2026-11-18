// POST /api/login with a form field "password": sets the login cookie
import { authToken } from "../_middleware.js";

export async function onRequestPost({ request, env }) {
  const form = await request.formData();
  const password = form.get("password") || "";

  // Compare hashes, not the raw strings, so the time taken doesn't reveal how much of the password was right
  if (!env.APP_PASSWORD || (await authToken(password)) !== (await authToken(env.APP_PASSWORD))) {
    return Response.redirect(new URL("/login?wrong=1", request.url), 302);
  }

  const token = await authToken(env.APP_PASSWORD);
  // Secure: only sent over https (always the case online; locally the app runs on http://localhost)
  const secure = new URL(request.url).protocol === "https:" ? "; Secure" : "";
  return new Response(null, {
    status: 302,
    headers: {
      Location: "/",
      // HttpOnly: page scripts can't read it; SameSite=Strict: not sent by other sites; 7 days
      "Set-Cookie": `auth=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800${secure}`,
    },
  });
}
