# Implementation and reproduction notes

## Gate

At diffusion step 5, the gate requests attention measurements at layers 0, 14, and 27 in the conditional generation branch. Classifier-free guidance branches are excluded.

- Generation Intent: layer-27 attention mass to all preceding text, divided by that mass plus layer-27 attention to the original image's ViT and VAE tokens. The denominator includes `1e-9` for numerical stability.
- Visual Fidelity: the average input-image ViT attention mass across layers 0, 14, and 27.
- Continue if either signal is strictly greater than its threshold; otherwise abort.
- After abort, the partial image is discarded and text reasoning resumes. Later image requests are evaluated independently by the gate. Text generation and successfully generated images each consume one reasoning round; skipped or gate-aborted images do not.
