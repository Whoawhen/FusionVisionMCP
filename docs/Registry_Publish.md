# Publishing FusionVisionMCP to the MCP Registry

A runbook for listing this server in the public [MCP Registry](https://registry.modelcontextprotocol.io).
Cutting a GitHub release and listing in the registry are **two separate actions** here, on purpose — see
[Why these are split](#why-these-are-split).

Everything below was verified against the registry's OpenAPI spec and the
`2025-12-11` server schema on 2026-09-16.

---

## Read this first: listing is close to permanent

**The registry has no delete endpoint.** The strongest retraction available is `PATCH
/v0.1/servers/{name}/status` with `status: "deleted"`, and the name
`io.github.Whoawhen/fusion-vision-mcp` stays claimed either way.

There is also **no "under construction", "beta" or "draft" state.** The lifecycle enum is exactly
three values:

| Status | Meaning |
|---|---|
| `active` | Normal, listed |
| `deprecated` | "Don't use this anymore" — carries an optional `statusMessage` (≤500 chars) |
| `deleted` | Hidden; the name remains claimed |

Publishing `active` and immediately marking it `deprecated` is possible but reads as *"this was real,
now abandon it"* rather than *"not finished yet"*, and it burns the version number. If the server
isn't ready, the right move is to cut the GitHub release and **not** run the registry workflow.

---

## Current state

- **No tags exist**, locally or on the remote — the 2026-09-16 history flattening dropped them all.
- **No GitHub releases exist**, so `README_DETAILED.md`'s "download the latest `.mcpb` from the
  Releases page" instruction currently points at an empty page.
- `server.json` has never been published; `fileSha256` is empty and is filled in by the workflow.
- `pyproject.toml`, `manifest.json` and `server.json` all read **0.8.2**.

---

## Step 1 — cut the GitHub release

This builds the `.mcpb` bundle and attaches it to a release. It does **not** touch the registry.

```powershell
# From a clean tree on main, with CI green
git tag v0.8.2
git push origin v0.8.2
```

`.github/workflows/release.yml` fires on `v*.*.*` and does the rest. Watch it:

```powershell
gh run watch --repo Whoawhen/FusionVisionMCP
gh release view v0.8.2 --repo Whoawhen/FusionVisionMCP
```

A tag with a suffix (`v0.8.2-rc.1`) is marked as a **prerelease**, so it won't become the "Latest"
release the README links to. Use that if you want the bundle downloadable without presenting it as
the recommended version.

**Stopping here is a perfectly good outcome.** It fixes the broken install link with zero public
registry exposure. Steps 2 and 3 are only for when you actually want to be listed.

---

## Step 2 — dry-run the registry publish

Validates `server.json`, downloads the released bundle, stamps its checksum, and authenticates —
then stops without publishing.

```powershell
gh workflow run registry-publish.yml --repo Whoawhen/FusionVisionMCP -f tag=v0.8.2 -f dry_run=true
gh run watch --repo Whoawhen/FusionVisionMCP
```

The workflow refuses to continue unless all of these hold:

- `description` is 1–100 characters. **This has been wrong before** — it sat at 231 characters, which
  would have been rejected at publish time. It is now 95.
- `fileSha256` is a 64-character lowercase hex digest (stamped from the release asset, not by hand).
- `server.json`'s `version` and `packages[0].version` both equal the tag with `v` stripped.
- `packages[0].identifier` references the same tag being published.

If any check fails, fix it on `main`, re-tag, and start again from Step 1.

---

## Step 3 — publish

Only after a clean dry run. **This is the irreversible one.**

```powershell
gh workflow run registry-publish.yml --repo Whoawhen/FusionVisionMCP -f tag=v0.8.2 -f dry_run=false
gh run watch --repo Whoawhen/FusionVisionMCP
```

Authentication is GitHub OIDC (`mcp-publisher login github-oidc`), which works because the workflow
requests `id-token: write` and the server name's `io.github.Whoawhen/` namespace matches the
repository owner. No secret or PAT is needed.

### Verify

```powershell
curl.exe -s "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.Whoawhen/fusion-vision-mcp"
```

Check that `status` is `active`, `version` is what you published, and the package `identifier`
resolves to a real download.

---

## Publishing a later version

Same three steps with the new tag. Before Step 1, make sure the version is bumped everywhere —
`bumpversion` is configured in `pyproject.toml` and already updates `manifest.json` and both
`server.json` version fields:

```powershell
uv run --with bump-my-version bump-my-version bump patch   # or minor / major
```

Then re-run `uv tool install --editable . --force ...`, since `pyproject.toml` changed. Registry
versions are immutable — you cannot overwrite `0.8.2` once it is published, only add `0.8.3`.

---

## If you need to retract

There is no unpublish. Your options, weakest to strongest:

```powershell
# Mark deprecated, with a reason users will see
gh api --method PATCH "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.Whoawhen/fusion-vision-mcp/status" `
  -f status=deprecated -f statusMessage="Superseded by v0.9.0; see the repository README."

# Or hide it entirely (the name stays claimed)
gh api --method PATCH "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.Whoawhen/fusion-vision-mcp/status" `
  -f status=deleted
```

A single version can be retracted instead of the whole server via
`/v0.1/servers/{name}/versions/{version}/status`.

---

## Why these are split

`release.yml` used to do both jobs: one `git tag` push built the bundle, cut the release, **and**
listed the server publicly. Given that listing is effectively permanent, that is too much to happen
as a side effect of tagging — particularly on a repository where the tags were being recreated from
scratch after a history rewrite.

So `release.yml` now only builds and releases, and `registry-publish.yml` is `workflow_dispatch`-only
with `dry_run` defaulting to **true**. You have to ask for the irreversible thing twice.

---

## Notes

- **`server.json`'s `status` field is vestigial.** The `2025-12-11` schema has no `status` property;
  lifecycle status is server-side metadata set through the status endpoint, not by the publisher. The
  line is left in place because it is harmless and was present in the configuration that has always
  been used — but do not expect editing it to change anything.
- **`description` is capped at 100 characters** and the schema asks for capabilities rather than
  implementation details. The model list belongs in the README, which `websiteUrl` points at.
- Validate a candidate `server.json` locally at any time against
  `https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json`, or POST it to the
  registry's `/v0.1/validate` endpoint.
