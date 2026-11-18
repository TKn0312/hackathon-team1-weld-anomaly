// POST /api/logout: deletes the login cookie and goes back to the login page
export async function onRequestPost({ request }) {
  return new Response(null, {
    status: 302,
    headers: {
      Location: new URL("/login", request.url).toString(),
      "Set-Cookie": "auth=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
    },
  });
}
