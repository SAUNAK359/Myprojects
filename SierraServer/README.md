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

## Secure quick start with Docker Desktop

```bash
cd SierraServer
cp .env.example .env
python scripts/generate_secrets.py
```

Copy the generated `SIERRA_DB_PASSWORD`, `SIERRA_JWT_SECRET`, and `SIERRA_PROVIDER_MASTER_KEY` values into your local `.env`. Never paste those values into GitHub, chat, screenshots, or issue trackers.

Start the database first:

```bash
docker compose up -d db
```

Build the API image and run migrations once:

```bash
docker compose build api
docker compose run --rm api alembic upgrade head
```

Create the first administrator securely. The password is read interactively, hashed immediately, and is never written to `.env`:

```bash
docker compose run --rm api python scripts/bootstrap_admin.py
```

Start Sierra Server:

```bash
docker compose up -d api
```

Open locally:

- API: `http://127.0.0.1:8080`
- API docs in development only: `http://127.0.0.1:8080/docs`
- health: `http://127.0.0.1:8080/health/ready`
- admin: `http://127.0.0.1:8080/admin/login`

Before public exposure, switch `SIERRA_ENV=production`, configure `SIERRA_PUBLIC_BASE_URL=https://...`, restart the API, and verify `/health/ready` locally.

## Public access

For customer devices outside your LAN, do **not** expose port `8080`, PostgreSQL, or Docker daemon ports directly to the internet. The preferred laptop deployment is an outbound-only Cloudflare Tunnel using the `cloudflared` Compose profile.

Keep the API host binding at `127.0.0.1:8080`. Configure the public tunnel hostname to proxy to `http://api:8080` on the Docker `edge` network. No home-router port forwarding is required.

For the administrator dashboard, prefer a Cloudflare Access policy or use the local `http://127.0.0.1:8080/admin/login` page from the server laptop. Do not publish an unprotected admin dashboard URL merely because API authentication exists.

## Provider setup

Provider credentials are never compiled into the APK. Sign into the admin dashboard/API and add OpenAI-compatible multimodal routes. Provider API keys are Fernet-encrypted before being written to PostgreSQL and are never returned to the browser.

Provider endpoints must use public HTTPS port 443. Local/private/link-local/reserved network targets are rejected to reduce SSRF risk.

## Security controls

- `.env` and runtime files are Git-ignored
- API binds to localhost on the host machine
- PostgreSQL is isolated on an internal Docker network
- API container runs as a non-root user
- passwords use Argon2
- refresh tokens are random and stored only as hashes
- provider API keys are encrypted at rest with a separate Fernet key
- authentication/job/admin endpoints have request-rate limits
- login errors are deliberately generic to reduce account enumeration
- provider egress rejects private/local network destinations
- uploaded analysis images have byte and pixel/decompression limits
- analysis frames and temporary context are removed after processing
- production OpenAPI/docs are disabled
- browser security headers and no-store headers protect auth/admin responses
- CI runs import/static/unit checks plus a dependency vulnerability audit

## Production rules

- keep `.env` outside Git
- use unique generated `JWT_SECRET`, `PROVIDER_MASTER_KEY`, and database password
- back up PostgreSQL; analysis images are temporary and should not be backed up
- use one stable HTTPS hostname for the APK
- configure Docker Desktop/tunnel auto-start on the laptop
- disable automatic sleep/hibernate while serving customers
- use a UPS if the laptop is intended to be the production origin
- periodically test database restore, not only backup creation
- rotate provider API keys when staff/access changes
- never copy production secrets into support logs

## Server vs APK responsibilities

**Server:** account authority, credits, cloud jobs, provider secrets/routing, admin monitoring.

**APK:** camera hardware, Sierra β, final capture, local validation/finish, gallery, privacy mode.
