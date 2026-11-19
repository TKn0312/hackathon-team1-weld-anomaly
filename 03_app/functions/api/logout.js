// POST /api/logout: deletes the session cookie and goes back to the login page
export async function onRequestPost({ request }) {
  return new Response(null, {
    status: 302,
    headers: {
      Location: new URL("/login", request.url).toString(),
      "Set-Cookie": "session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
    },
  });
}
