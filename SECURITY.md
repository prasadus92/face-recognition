# Security Policy

## Supported versions

Only the latest released minor version receives security fixes. Older versions may receive a backport at maintainer discretion but are otherwise unsupported.

| Version | Supported |
|---------|-----------|
| 2.1.x   | ✅        |
| 2.0.x   | ❌        |
| 1.x     | ❌        |

## Reporting a vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Report them privately using one of the following channels:

1. **GitHub Security Advisories** — preferred. Open a new advisory at <https://github.com/prasadus92/face-recognition/security/advisories/new>.
2. **Email** — `prasadus92@gmail.com`. Please include:
   - A description of the vulnerability and its impact.
   - Step-by-step reproduction instructions or a proof-of-concept.
   - The commit SHA / release version you tested against.
   - Whether you'd like to be credited in the advisory.

You will receive an acknowledgement within **72 hours**. I'll work with you on triage, fix, disclosure timeline, and credit.

## Disclosure process

1. Report received and acknowledged (≤ 72h).
2. Triage + reproduction (≤ 1 week).
3. Fix developed on a private branch.
4. Coordinated disclosure — advisory published and fixed release tagged. Reporter is credited unless they prefer anonymity.

## Scope

In scope:

- The Java library (`com.facerecognition.*`) and its REST / CLI surfaces.
- The published Docker image (once available).
- CI configuration in `.github/workflows/`.

Out of scope:

- Misconfiguration of a downstream deployment (e.g. exposing the API to the public internet without auth).
- Vulnerabilities in transitive dependencies that are not exploitable through this project's usage — please report those upstream.
- Findings from automated scanners without a reproducer.

## Hardening notes for operators

- Run the service behind an authenticating reverse proxy or API gateway, or enable the built-in API-key filter (`facerecognition.security.api-key`).
- Keep the default rate limiter enabled (`facerecognition.ratelimit.enabled: true`).
- Restrict `/actuator/*` exposure in production (`management.endpoints.web.exposure.include`).
- Pin and regularly update the base image if you build your own variant of the Dockerfile.
- Store serialized `TrainedModel` files in a trusted location — the persistence layer uses Java serialization, so never import a model you did not generate yourself.
