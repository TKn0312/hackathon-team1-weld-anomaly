# 04_final/app · the final app

The final app's code is in **[`03_app/`](../../03_app/)**: we kept building there instead of copying it, so there is one version of the code.

| | |
|---|---|
| Live | **https://hackathon-team1-weld-app.pages.dev** |
| Page | [`03_app/public/`](../../03_app/public/) |
| Worker (API) | [`03_app/functions/`](../../03_app/functions/) and [`03_app/lib/`](../../03_app/lib/) (accounts, signed links) |
| Database | [`03_app/schema.sql`](../../03_app/schema.sql), [`03_app/migrations/0002_users.sql`](../../03_app/migrations/0002_users.sql) |
| Settings | [`03_app/wrangler.toml`](../../03_app/wrangler.toml) (no secrets: the password is a Cloudflare secret) |
| How to run and deploy | [`03_app/README.md`](../../03_app/README.md) |

What changed on Thursday: user accounts behind the team password (each user sees only their own checks), and "Look inside the network →" opens NEURAL / LAB with the same X-ray.
