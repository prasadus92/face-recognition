# Security policy

## Supported versions

Security fixes land on `main` and in the next release. Older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue. Email `prasadus92@gmail.com` with:

- what the problem is and what an attacker could do with it;
- steps or a proof of concept to reproduce it;
- the commit or release you tested;
- whether you would like to be credited.

I will acknowledge the report within 72 hours, agree a fix and disclosure date with you, and credit you in the release notes unless you prefer otherwise.

## Scope

In scope: the library (`com.facerecognition.*`), its REST API and CLI, the Dockerfile, and the CI workflows in `.github/workflows/`.

Out of scope: insecure deployments (for example an API exposed to the internet without authentication), dependency vulnerabilities that this project cannot trigger (please report those upstream), and scanner output without a reproducer.

## Hardening notes for operators

- Put the API behind an authenticating proxy, or set `facerecognition.security.api-key` to require an `X-API-Key` header.
- Keep the rate limiter on (`facerecognition.ratelimit.enabled: true`).
- Limit `/actuator/*` exposure in production through `management.endpoints.web.exposure.include`.
- Model files contain the enrolled face crops. Store them as you would any biometric data.
- `POST /api/v1/model/import` decodes uploads behind an allowlist of the classes a model contains, with size limits. Still, only import model files from a source you trust.
