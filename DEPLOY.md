# Deploying to GitHub Pages — Recipe

This site is a fully static SvelteKit app (`@sveltejs/adapter-static`) published to
GitHub Pages at **https://sedgewickmm18.github.io/mamba2-explainer/** using the
[`gh-pages`](https://www.npmjs.com/package/gh-pages) npm package.

## The recipe

```bash
npm run deploy
```

That's it. It runs `predeploy` (`npm run build`) automatically, then pushes the
`build/` directory to the `gh-pages` branch of `origin`. GitHub Pages serves that
branch. Allow a minute or two for the CDN to pick up the new files.

## One-time setup (already done in this repo)

GitHub Pages must be told where the site lives. In the repo on GitHub:
**Settings → Pages → Build and deployment → Branch: `gh-pages`, folder: `/ (root)` → Save.**

## How the pieces fit (and what each file does)

| File | Role |
|------|------|
| `svelte.config.js` | `adapter-static` writes the static site to `build/` with a `404.html` fallback. `paths.base` is set to `/mamba2-explainer` for anything that isn't `vite dev` (see below). |
| `package.json` | `"deploy": "gh-pages -d build --nojekyll"` — publishes `build/` (not `dist/`!) and disables Jekyll processing (see gotcha #2). |
| `static/.nojekyll` | Empty marker file, copied into `build/` by the adapter. Belt-and-braces so the published directory always disables Jekyll, no matter how it gets uploaded. |
| `src/routes/+page.svelte` | Calls `initLoader(base)` with `base` from `$app/paths`, so data is fetched from `/mamba2-explainer/data/...` in production and `/data/...` in dev. |
| `.gitignore` | `build/`, `dist/`, `.svelte-kit/` and `node_modules/` are build artifacts — they are deployed, never committed. |

## The three gotchas that break this setup (all were live bugs once)

1. **Wrong output directory.** The adapter writes to `build/`; the deploy script
   must publish `build/`. An earlier version published a stale `dist/` folder, so
   deploys silently shipped an old build. If you ever rename the adapter output
   (`pages:`/`assets:` in `svelte.config.js`), update `-d` in the deploy script to match.

2. **Jekyll eats `_app/`.** GitHub Pages builds sites with Jekyll unless told not
   to, and Jekyll ignores files/directories starting with an underscore — which is
   exactly where SvelteKit puts all JS/CSS (`build/_app/`). Symptom: the page loads
   but is blank, and every `/_app/...` request 404s. The `--nojekyll` flag (plus
   `static/.nojekyll`) prevents this.

3. **Project pages live under a sub-path.** The site is served from
   `/mamba2-explainer/`, not `/`. Two things must respect that:
   - `paths.base` in `svelte.config.js` so generated asset/route URLs are prefixed.
   - Any runtime `fetch()` of static data must use `base` from `$app/paths` — an
     absolute `fetch('/data/manifest.json')` hits `https://<user>.github.io/data/...`
     and 404s. (If the repo is ever renamed or moved to `<user>.github.io` root,
     update the base in `svelte.config.js` — set it to `''` for a user-site.)

## Verifying a deployment locally before pushing

```bash
npm run build
npm run preview
```

Preview serves the production build at `http://localhost:4173/mamba2-explainer/`
(the base path is included — that's a good sign it's baked in correctly). Check that:

- the page renders,
- the network tab shows `data/manifest.json` loading from under `/mamba2-explainer/`,
- a bogus deep URL (e.g. `/mamba2-explainer/no/such/page`) still renders the app via the 404 fallback.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Blank page, `/_app/...` requests 404 on the live site | Missing `.nojekyll` — deploy with `--nojekyll`, keep `static/.nojekyll`. |
| Data never loads ("Could not load manifest") | Data fetched with an absolute path — use `base` from `$app/paths` (see `+page.svelte`). |
| Styles/assets 404 right after deploy | Old `dist/`-style stale artifact published, or base path mismatch — confirm `gh-pages -d build` and `paths.base`. |
| Deploy succeeds but site unchanged | CDN cache — wait 1–2 minutes and hard-refresh; also check GitHub → Settings → Pages for build errors and that it serves `gh-pages` / `(root)`. |
| 404 on the site root | Pages not enabled or wrong branch/folder in Settings → Pages. |
| `gh-pages` errors about a dirty `gh-pages` cache | Remove the package's local cache: `rm -rf node_modules/gh-pages/.cache`, then re-run. |

## Notes

- `npm run deploy` only updates the `gh-pages` branch — always commit and push
  `main` separately so source and site stay in sync.
- The stale `dist/` directory (from the earlier broken setup) can be deleted
  safely; nothing references it anymore.
- Alternative: GitHub Actions can build and publish on every push to `main`
  (with `actions/deploy-pages`), removing the manual step — the current
  `gh-pages`-package setup was chosen to keep it simple.
