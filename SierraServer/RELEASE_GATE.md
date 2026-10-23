# Sierra Vision Server — Release Gate

This branch is intentionally separate from the released Android branch.

A server release is acceptable only when all of the following are true:

- FastAPI package imports cleanly.
- Server unit tests pass.
- Server static correctness checks pass.
- Android integration compiles, unit-tests, and assembles with an HTTPS `SIERRA_SERVER_BASE_URL` build property.
- No `.env`, provider API key, JWT signing secret, database password, refresh token, or customer password is committed or baked into the APK.
- PostgreSQL migrations apply cleanly from an empty database.
- New accounts receive exactly five trial credits by server-side ledger mutation.
- Creating a Sierra AI job reserves one credit atomically.
- Provider failure, user cancellation, restart recovery, and ready-job expiry release the reservation.
- A credit is consumed only after the APK confirms a successfully saved final Sierra AI photograph.
- Repeated job creation/completion is idempotent and cannot double-charge.
- Sierra beta remains available when the server, provider, network, account, or credits are unavailable.
- Analysis frames and temporary camera context are erased after inference/cancellation/recovery.
- Provider/model identities and provider credentials remain server/admin concerns, not customer API/UI details.
- Production startup refuses default secrets, a default DB password, or a non-HTTPS public base URL.
- The API is exposed through HTTPS; PostgreSQL is not publicly exposed.
- Account password change and authenticated account deletion are available.

Payment/subscription gateway integration is intentionally not faked in this phase: the authoritative credit/entitlement layer is implemented, while the commercial payment provider will be integrated separately once selected.
