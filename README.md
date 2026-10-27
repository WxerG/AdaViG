# AdaViG project page

A static academic project page for **Model Guides You How to Draw: Adaptive Visual Gating for Unified Multimodal Reasoning**. No build step, framework, or package installation is required.

## Preview

```bash
python3 -m http.server 8080
```

Open http://localhost:8080. Relative asset paths support both a project repository (`username.github.io/AdaViG/`) and an organization site (`adavig.github.io/`).

## Repository layout

Code: https://github.com/Xer-GWX/AdaViG/tree/main

This page: https://github.com/Xer-GWX/AdaViG/tree/gh-pages

Intended project URL: https://xer-gwx.github.io/AdaViG/

## Publish on GitHub Pages

1. Push this directory to the existing repository's `gh-pages` branch. Include the hidden `.github/` directory.
2. In **Settings → Pages → Build and deployment**, select **GitHub Actions**.
3. Run the **Deploy project page** workflow, or push another commit to `gh-pages`.

The workflow publishes only `index.html`, `site-config.js`, `.nojekyll`, and `assets/`. GitHub will show the actual public URL after deployment. A successful source push alone does not confirm a live Pages deployment.

For the exact `https://adavig.github.io/` address, the owner must control the GitHub account or organization named `adavig` and publish from its `adavig.github.io` repository. Otherwise use a project page under the account you control.

## Content updates

Edit `site-config.js` for authors, affiliations, paper URL, code URL, and BibTeX. Names and affiliations were taken from `UMMs_cot/myarxiv.tex`. No acceptance venue or publication identifier is assumed. Without a paper URL, the hero links to the overview; without a code URL, the buttons download the included release archive.

The citation uses `@unpublished` and omits an unconfirmed year/venue/identifier. Replace it with the final bibliographic entry once available.

Reported results are transcribed from the manuscript's active main-results table in `assets/main.js`. They describe paper experiments, not a new reproduction. Model toggles change the five benchmark rows and the reported averages. The method slider is an illustrative gate, not model inference.

## Assets

Paper figures are rasterized from these local originals:

| Web asset | Paper figure |
| --- | --- |
| `teaser.png` | `intro2.pdf` |
| `overview.png` | `m2_2.pdf` |
| `efficiency.png` | `fig_cost_real5.pdf` |
| `case-study.png` | `fig_case_study4.pdf` |

Figure buttons open a native accessible dialog. Navigation, keyboard controls, reduced-motion support, and responsive layouts are included. Google Fonts are optional; local system fallbacks keep the page usable without a font connection. All figures, code download, styles, and scripts are served locally.

The bundled `assets/AdaViG-code.zip` is a snapshot of the prepared code release; update it or set `codeUrl` when preparing a newer release. It includes the code's license and provenance. Paper figures remain the authors' research material. The layout was inspired by https://mllm-dmlr.github.io/; its source and assets were not copied.
