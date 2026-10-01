import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter({
            pages: 'dist',
            assets: 'dist',
            fallback: undefined,
            precompress: false,
            strict: true
        }),
        // Set base path for GitHub Pages deployment.
        // Change this to your repo path when deploying.
        // e.g. paths: { base: '/llama.cpp/mamba2-explainer' }
        paths: {
            base: process.env.BASE_PATH || ''
        }
    }
};

export default config;
