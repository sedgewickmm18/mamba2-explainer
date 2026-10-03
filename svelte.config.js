import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter({
            pages: 'build',
            assets: 'build',
            fallback: '404.html', // Highly recommended for routing on GitHub Pages
            precompress: false,
            strict: true
        }),
        // Base path for GitHub Pages project sites: the site is served from
        // https://<user>.github.io/mamba2-explainer/, so production builds must
        // prefix all assets and routes with the repo name. `vite dev` serves at
        // the root, so it gets an empty base.
        paths: {
            base: process.argv.includes('dev') ? '' : '/mamba2-explainer'
        }
    }
};

export default config;
