# Security Policy

## Status

DimensionalBase is alpha software (0.x). The security surface is evolving and
not all hardening work is complete. Do not deploy untrusted-network-facing
instances without a reverse proxy that adds TLS, authentication enforcement,
and rate limiting.

## Supported Versions

Only the latest `0.x` minor receives security fixes during the alpha. Once a
`1.0` is tagged we will publish a formal support window here.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security reports.**

Please email **security@dimensionalbase.dev** with:

- A description of the issue and its impact
- Steps to reproduce (a minimal PoC is ideal)
- The affected version / commit SHA
- Any suggested mitigation

You can expect:

- An acknowledgement within **3 business days**
- A triage update within **7 business days**
- A coordinated disclosure timeline once severity is confirmed

If you do not get a response within a week, please open a GitHub issue
titled "Security report awaiting response" (without details) so we can
re-engage.

## Scope

In scope:
- The `dimensionalbase` Python package
- The FastAPI server (`dimensionalbase.server`)
- The MCP server (`dimensionalbase.mcp`)
- The CLI (`dimensionalbase.cli`)

Out of scope:
- Vulnerabilities that require local code execution as the same user
  running the process (e.g., reading the SQLite file you own)
- Issues in optional third-party dependencies (please report upstream)
- Denial of service via unbounded memory in development-mode configs;
  production deployments are expected to front the server with a proxy

## Known Limitations (current alpha)

These are tracked and being addressed; do not file as new vulnerabilities:

- API-key revocation is per-process; multi-worker deployments may serve a
  revoked key until the cache TTL expires
- The default server bind has been changed to `127.0.0.1`; binding to
  public interfaces requires `--host 0.0.0.0` and is your responsibility
  to secure
- Encryption-at-rest passphrases use PBKDF2-HMAC-SHA256 with per-record
  random salts; legacy data written with the old static-salt format is still
  readable, but operators should plan migrations before changing encryption
  material

## Disclosure Credit

Reporters who follow coordinated disclosure are credited in the release
notes (with permission).
