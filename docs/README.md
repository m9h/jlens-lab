# Tutorials

Written from failures we actually shipped. Every number here came from a run.

1. **[What the Jacobian lens actually computes](01-what-the-jacobian-lens-computes.md)**
   The derivative everyone gets wrong, the steelman that's correct (it's a tuned lens),
   and the randomization control that shows it isn't trivial anyway.

2. **[Fitting a lens without fooling yourself](02-fitting-without-fooling-yourself.md)**
   `jlens.fit()` has no stopping rule. Anthropic's own lenses used 454–775 prompts.
   `~100` is their floor, not their fit size. Under-fitting is silent.

3. **[Reading a lens honestly](03-reading-a-lens-honestly.md)**
   Three controls. The logit-lens floor, the randomization test, the distance-only null.
   Plus: the paper's `pass@k` metric rewards noise, and here's the proof.

4. **[Lensing things that aren't transformers](04-lensing-things-that-arent-transformers.md)**
   The workspace is clearest in the models `jlens` can't load. Layouts, the 73GB-per-layer
   Mamba wall, and why fused kernels are mandatory.
