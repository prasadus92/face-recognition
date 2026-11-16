# Security policy

## Supported versions

Security fixes land on `main` and in the next release. Older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue. Email `prasadus92@gmail.com` with:

- what the problem is and what an attacker could do with it;
- steps or a proof of concept to reproduce it;
- the commit or release you tested;
- whether you would like to be credited.

I will acknowledge the report within 72 hours, agree on a fix and a disclosure date with you, and credit you in the release notes unless you prefer otherwise.

## Scope

In scope: the library (`com.facerecognition.*`), its REST API and CLI, the Dockerfile, and the CI workflows in `.github/workflows/`.

Out of scope: insecure deployments (for example an API exposed to the internet without authentication), dependency vulnerabilities that this project cannot trigger (please report those upstream), and scanner output without a reproducer.

## Hardening notes for operators

- Put the API behind an authenticating proxy, or set `facerecognition.security.api-key` to require an `X-API-Key` header.
- Keep the rate limiter on (`facerecognition.ratelimit.enabled: true`). It keys on the client address. Tomcat takes that address from `X-Forwarded-For` only when the connection comes from a trusted proxy, by default any address in `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `100.64.0.0/10`, `169.254.0.0/16`, `127.0.0.0/8`, `fc00::/7`, `fe80::/10` or `::1`. If clients can reach the service directly from those ranges (a LAN, a VPC, Tailscale, Docker Desktop), set `server.tomcat.remoteip.internal-proxies` to your proxy's exact address. Only `X-Forwarded-For` is read; the RFC 7239 `Forwarded` header and `X-Forwarded-Prefix` are not.
- Limit `/actuator/*` exposure in production through `management.endpoints.web.exposure.include`.
- Model files contain the enrolled face crops. Store them as you would any biometric data.
- `POST /api/v1/model/import` decodes uploads behind an allowlist of the classes a model contains, with size limits. Still, only import model files from a source you trust.
