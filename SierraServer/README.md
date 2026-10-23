# Sierra Vision Server

Laptop-hosted control plane for Sierra Vision.

## What this service owns

- customer signup/login and session refresh
- password hashing and account lockout
- 5-credit trial grant for every new account
- authoritative credit reservation/commit/release ledger
- durable Sierra AI jobs with idempotency
- temporary analysis-image handling (deleted after inference)
- multi-provider OpenAI-compatible VLM routing and health scoring
- strict Sierra photography-plan validation
- provider secrets encrypted at rest
- admin monitoring/control APIs and local dashboard
- restart recovery for queued/processing jobs

The Android APK remains useful without this service through Sierra β. Cloud credits are never authoritative on-device.

## Credit lifecycle

`available = balance - reserved`

1. APK submits a reduced analysis frame with an idempotency key.
2. Server atomically checks `available >= 1` and reserves one credit.
3. A durable job is queued and processed by the server worker.
4. Provider returns a validated Sierra photography plan.
5. Job enters `ready`; the reservation remains held.
6. APK executes the plan locally and captures the final full-quality photograph.
7. APK calls `POST /v1/jobs/{job_id}/complete`.
8. Server atomically converts the reservation into one consumed credit.
9. Failure/cancel/expiry releases the reservation instead.

A provider response alone never consumes a credit.

## Quick start (Windows/Linux/macOS with Docker Desktop)

```bash
cd SierraServer
cp .env.example .env
python scripts/generate_secrets.py
# copy the generated secrets into .env, then set admin credentials

docker compose up --build -d
```

Open:

- API: `http://localhost:8080`
- API docs: `http://localhost:8080/docs`
- health: `http://localhost:8080/health/ready`
- admin login: `http://localhost:8080/admin/login`

For customer devices outside your LAN, do **not** expose port 8080 directly. Put the server behind HTTPS (Caddy profile, a secure tunnel, or another reverse proxy) and set the APK API base URL to that HTTPS origin.

## Provider setup

Provider credentials are not compiled into the APK. Sign into the admin dashboard/API and add one or more OpenAI-compatible multimodal routes (Groq/OpenRouter/NVIDIA/Gemini OpenAI-compatible endpoints or another compatible service). The API key is encrypted before it is written to PostgreSQL.

## Production rules

- keep `.env` outside Git
- use unique `JWT_SECRET` and `PROVIDER_MASTER_KEY`
- run PostgreSQL on local/private networking only
- expose the API only through HTTPS
- back up PostgreSQL, not temporary analysis images
- use one stable public hostname for the APK
- configure router/firewall/tunnel auto-start on the laptop
- use a UPS if this laptop is intended to be the production origin

## Server vs APK responsibilities

**Server:** account authority, credits, cloud jobs, provider secrets/routing, admin monitoring.

**APK:** camera hardware, Sierra β, final capture, local validation/finish, gallery, privacy mode.
