// Runs before EVERY request: the login gate.
// A visitor without a valid session cookie only gets the login page and the login API.
// The session cookie names ONE user (see lib/users.js); the functions use it to show only that user's data.
import { userFromCookie } from "../lib/users.js";

// Pages serves login.html at /login (it drops the .html)
const PUBLIC_PATHS = ["/login", "/login.html", "/api/login", "/style.css"];

export async function onRequest(context) {
  const { request, env, next } = context;
  const url = new URL(request.url);

  if (PUBLIC_PATHS.includes(url.pathname)) {
    return next();
  }
  // Signed one-image links for NEURAL / LAB: no cookie, but the function itself checks the signature
  if (url.pathname.startsWith("/api/shared/")) {
    return next();
  }

  const userId = await userFromCookie(env, request.headers.get("Cookie"));
  if (userId) {
    context.data.userId = userId;   // logged in: the functions read it as data.userId
    return next();
  }

  if (url.pathname.startsWith("/api/")) {
    return Response.json({ error: "Not logged in" }, { status: 401 });
  }
  return Response.redirect(new URL("/login", url), 302);
}
