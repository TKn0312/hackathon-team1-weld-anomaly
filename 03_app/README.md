# Weld X-ray Check: the app (Wednesday)

**Public URL: [hackathon-team1-weld-app.pages.dev](https://hackathon-team1-weld-app.pages.dev)**

A private web app for screening weld X-rays. It uses our own autoencoder through our model server on Render (`02_model_server/`) and stores every check in a database. Built by Amelia from the class example [`cloudflare_app/`](https://github.com/iaac-maai/ai-methods-1-2026/tree/main/hackathon/docs/cloudflare_app).

## Log in

The app is protected by **one team password**. The password is given in the course hand-in, **not in this repository**.
1. Open the URL: you land on the login page.
2. Enter the password. The server compares it with a Cloudflare secret (`APP_PASSWORD`) and sets a login cookie for 7 days. The cookie stores a hash of the password, never the password itself.
3. **Log out** (top right) deletes the cookie.

Without the cookie, every page sends you to the login page and every `/api/...` call answers `401 Not logged in`.

## What the app does

1. **Upload an X-ray tile of a weld** (PNG or JPEG, up to 5 MB) and press **Check tile**.
2. The model, trained only on good welds, tries to redraw it. The app shows:
   - the **risk level**:
     - **LOW**: looks like the good welds it learned from
     - **CHECK**: somewhat unusual, have a look
     - **HIGH**: very unlike a good weld, inspect it
   - the **score**, marked on a scale between the two thresholds (LOW up to 0.000255, HIGH above 0.001442)
   - **three images side by side**: the uploaded X-ray, the model's redraw (how a good weld would look) and the **error heat map** (bright = where the redraw failed = where to look)
3. **Recent checks:** a table of the last 20 checks, read from the database. Click a row to see the saved X-ray and its error map again.
4. **Look inside the network →** opens [NEURAL / LAB](https://hackathon-team1-weld-anomaly.onrender.com/#architecture), our interactive explainer of the same model.

**A screening aid, not a verdict:** a qualified inspector decides.

**The model is waking up.** The free Render server sleeps after 15 minutes without requests, and the first call then takes about a minute. The app does not show an error. It shows *"Model is waking up, about 1 minute"* and retries by itself every 10 s for up to 2 minutes.

## How it works

```
Browser ──► Cloudflare Pages Functions ──► Render: our model server (/gradio, /predict)
 (page)       password check                   the autoencoder (ONNX) scores the tile
              save the result  ──► D1  (one row per check)
              save the images  ──► R2  (the X-ray and its error map)
```

- **The page never calls the model server itself.** It only calls the app's own functions (`/api/...`). They check the password, then forward the image to the model server.
- **No new model API:** the functions call **the same `/predict` API** that we deployed on Tuesday (`MODEL_URL` in `wrangler.toml`). The model only runs on Render.
- **NEURAL / LAB gives the same scores.** It is our other app on the same Render server. For our 4 example tiles, both apps return the same scores:

  | Tile | Score |
  |---|---|
  | crack | 0.004769 |
  | lack of penetration | 0.010362 |
  | no defect | 0.000173 |
  | porosity | 0.011129 |

| Address (all behind the password) | What it does |
|---|---|
| `POST /api/login`, `POST /api/logout` | Sets or deletes the login cookie |
| `GET /api/status` | Asks the model server if it is awake (it answers when it is ready) |
| `POST /api/predict` | Checks the file, sends it to the model server, stores the images in R2 and one row in D1, returns the result |
| `GET /api/history` | The 20 most recent checks from D1 |
| `GET /api/images/<id>` (`?kind=error`) | A saved X-ray (or its error map) from R2 |

## Cloudflare services and why

| Service | Used for | Why |
|---|---|---|
| **Pages** | Hosts the page (`public/`) at a public `pages.dev` address | Free, fast everywhere, HTTPS included, and it does not sleep |
| **Pages Functions** (`functions/`) | The server code: the password gate, the call to the model server, saving the results | The password must be checked on a server, not in the browser. The functions also keep the model server address and the secret out of the page. |
| **D1** (database `hackathon-team1-weld-db`) | One row per check: id, date and time, file name, score, threshold, anomaly yes/no, risk level, score ÷ threshold | The results stay after the page closes, and everyone on the team sees the same history. D1 is SQL, so the history is one query. |
| **R2** (bucket `hackathon-team1-weld-images`) | The uploaded X-ray and the error map the model returned for it | A database is the wrong place for images. R2 stores files, and the history can show them again. |
| **Secrets** (`APP_PASSWORD`) | The team password | Never in the code or on GitHub. Locally it lives in `.dev.vars`, which is git-ignored. |

The D1 table (`schema.sql`):

| Column | Meaning |
|---|---|
| `id` | A random id (`crypto.randomUUID`) |
| `created_at` | Date and time (UTC) |
| `image_key` | Where the X-ray is stored in R2 |
| `file_name` | The name of the uploaded file |
| `score` | The anomaly score from the model server |
| `threshold` | The 95% threshold used |
| `is_anomaly` | 1 = above the threshold, 0 = below |
| `risk_level` | `low` / `check` / `high`, as decided by the model server |
| `score_vs_threshold` | The score as a multiple of the threshold |

**Safety details:**
- **Uploads:** only real PNG and JPEG files are accepted. The server checks the file's first bytes, not its name.
- **Database:** user input never goes into the SQL text (`?` placeholders).
- **Images:** saved images are served only as images.
- **Errors:** error details go to the Cloudflare logs, not to the browser.

## Design (Miro) compared with the app

On our Miro board, exported to [diagrams/](diagrams/):
- **2 user stories:** (1) a weld inspector checks a weld; (2) a quality manager reviews past checks, which only the team can see.
- **1 system diagram:** four parts: page → Pages Functions → model server on Render, and Functions → D1 + R2.
- **2–3 sequence diagrams**, one per user story.

**What is different, and why:**
- **The wake-up needed more than the story says.** The page retries every 10 s for up to 2 minutes. An extra `/api/status` request keeps Render starting.
- **A saved check shows 2 images, a new check shows 3.** R2 stores only the X-ray and the error map, not the model's redraw. The error map is what an inspector needs, and storing fewer images keeps R2 smaller.
- **Story 2 cannot say *who* made a check.** One shared team password means no separate users, so D1 has no user column. Separate users would need a real login, such as Sign in with Google.
- **One link the diagrams leave out:** the "Look inside the network" button links to NEURAL / LAB, which explains the same model.

## Run it locally

Needs [Node.js](https://nodejs.org) (LTS). From this folder:

```bash
npm install                              # installs wrangler (Cloudflare's tool)
echo APP_PASSWORD=choose-one > .dev.vars # the local password; .dev.vars is never committed
npm run db:local                         # creates the D1 table in a local copy of the database
npm run dev                              # http://localhost:8788
```

In `wrangler.toml`, `MOCK_MODEL = "1"` returns a made-up answer instead of calling Render (to work on the page offline). It is `"0"` for the real app.

**Put it online** (once per Cloudflare account; Amelia did this for the team):

```bash
npx wrangler d1 create hackathon-team1-weld-db       # then paste its id into wrangler.toml
npx wrangler r2 bucket create hackathon-team1-weld-images
npm run db:remote                                    # creates the table in the online database
npx wrangler pages secret put APP_PASSWORD           # the online password
npm run deploy                                       # publishes to hackathon-team1-weld-app.pages.dev
```

## Files

| File | Contents |
|---|---|
| `public/index.html`, `login.html`, `style.css` | The page and the login page |
| `functions/_middleware.js` | The password gate: runs before every request |
| `functions/api/login.js`, `logout.js` | Log in and out |
| `functions/api/status.js` | Is the model server awake? |
| `functions/api/predict.js` | Model call, wake-up handling, saving to R2 and D1 |
| `functions/api/history.js` | The recent checks |
| `functions/api/images/[id].js` | Saved images from R2 |
| `schema.sql` | The D1 table |
| `wrangler.toml` | Project name, `MODEL_URL`, the D1 and R2 names (no secrets) |
| `package.json` | Wrangler and the commands above |
| `diagrams/` | The Miro exports |

## Limitations

- **The first check after 15 idle minutes is slow** (about a minute) because the free Render server sleeps.
- **One shared password** for the whole team: no separate users, so the history does not show who made a check.
- **Not for whole radiographs:** the model expects tiles like the RIAWELC ones (about 227 × 227 px). A larger image is shrunk to 64 × 64 px and loses detail.
- **Free plan limits** for Pages Functions, D1 and R2 are far above what a class demo needs.
