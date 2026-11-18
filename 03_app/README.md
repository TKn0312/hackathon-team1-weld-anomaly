# 03_app · Weld X-ray Check

**Live app:** https://hackathon-team1-weld-app.pages.dev

A password-protected web app for weld inspectors. You upload an X-ray tile of a weld; our model (a convolutional autoencoder that learned only from good welds) tries to redraw it. Where it can't, the error map lights up. The app shows a risk level (**LOW / CHECK / HIGH**), the score against the two thresholds, the uploaded X-ray and the error heat map, and saves every check so it is still there after a reload. A button opens **NEURAL / LAB**, where you can look inside the same network.

It is a screening aid, not a verdict: a qualified inspector decides.

## How to log in

The app has one shared team password. It is given in the course hand-in, **not** in this repository. It is stored as a Cloudflare secret (`APP_PASSWORD`) and checked on the server, never in the page.

> The model server on Render sleeps after 15 minutes without use. The first check after a pause can take about a minute: the app shows "Model is waking up, about 1 minute…" and retries by itself, without showing an error.

## How it works

| Part | Service | What it does |
|---|---|---|
| **Page** | Cloudflare Pages (`public/`) | Plain HTML, CSS and JavaScript: the login page, upload, result and recent checks |
| **Worker** | Pages Functions (`functions/`) | `_middleware.js` checks the login cookie on every request. `/api/login`, `/api/logout`, `/api/predict` (checks the file, calls the model, saves the result), `/api/history` (last 20 checks), `/api/images/{id}` (an image from R2), `/api/status` (is Render awake?) |
| **Model server** | Render | Our FastAPI + Gradio server (`02_model_server/`). The Worker calls its `/predict` API at `MODEL_URL` with the image as base64 and gets back the score, risk level, redraw and error map |
| **Database** | Cloudflare D1 (`hackathon-team1-weld-db`) | Table `predictions`: one row per check (`id`, `created_at`, `file_name`, `score`, `threshold`, `is_anomaly`, `risk_level`, `score_vs_threshold`, `image_key`). See `schema.sql` |
| **File storage** | Cloudflare R2 (`hackathon-team1-weld-images`) | The uploaded X-ray (`uploads/{id}`) and its error heat map (`errors/{id}`). D1 keeps the key; R2 keeps the file |
| **Password** | Cloudflare secret | `APP_PASSWORD`, set with `npx wrangler pages secret put APP_PASSWORD` |

**Why these services:** Render runs our Python model; Cloudflare runs the small, fast server code next to the page and has a database (D1) and file storage (R2) built in. Only the Worker talks to the model, the database and the files, so the password and the model address stay on the server.

**Safety:** the Worker only accepts real PNG or JPEG files (it checks the file's first bytes, not its name), up to 5 MB. User input never goes into SQL text (`?` placeholders).

## Diagrams

In [`diagrams/`](diagrams) and on our Miro board:

- 2 user stories (the inspector, the engineer)
- 1 system diagram
- the D1 table with example rows
- 3 sequence diagrams: checking a weld, looking inside the network, the wake-up

## Our diagrams compared with the app we built

- **Story 1 (inspect and save) matches the app:** upload → model → R2 + D1 → result, and the history comes back from D1 after a reload.
- **Story 2 (look inside the network) only partly matches:** the button opens NEURAL / LAB, but the flagged weld is not passed along, so the engineer uploads it again there. The two apps run on different services (Cloudflare and Render) and do not share files.
- **Pages, not Workers:** the brief suggests Workers; we used Cloudflare Pages + Pages Functions (the class example), which do the same job.
- **The wake-up needed more than a simple retry:** the Worker first asks Render whether it is awake, and the page keeps one long "are you up?" request open while it retries every 10 s, because the first answer after a restart took about 47 s.
- **Saved checks show two images, not three:** the redraw is not stored in R2, only the X-ray and the error map.

## Run it locally

```bash
cd 03_app
npm install
echo "APP_PASSWORD=choose-one" > .dev.vars        # never commit .dev.vars
npx wrangler d1 execute hackathon-team1-weld-db --local --file=schema.sql
npx wrangler pages dev
```

`MOCK_MODEL = "1"` in `wrangler.toml` returns a made-up answer, so the page can be built without the model server; `"0"` calls the real one.

## Limits

- The model was trained on 227 × 227 px weld tiles from RIAWELC. A whole photo or a picture of something else gets a high score because it is unlike anything the model has seen, not because of a defect.
- Free plans: Render sleeps after 15 minutes; Cloudflare's free limits are enough for this project.
- Dataset: RIAWELC (Totino, Spagnolo & Perri, ICMECE 2022; Perri et al., Manufacturing Letters). No RIAWELC images are stored in this repository.
