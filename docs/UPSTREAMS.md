# Source selection and reproducibility

This is **not** a collection of Git Submodules. The three top-level directories contain copied, selected source files from pinned upstream commits, including CUDA extensions used by 2D-SuGaR.

- VGGT: \`vggt\` package, \`demo_colmap.py\`, install requirements, license. Omit original training system and example media.
- gsplat (3DGS): \`gsplat\` package + CUDA kernels and the COLMAP training example. Omit benchmarks, docs, test data, and unrelated examples.
- 2D-SuGaR: surface optimization/extraction, 2DGS runtime, 3DGS support, depth/normal prior generator, all **three** required CUDA extensions. Omit viewer/fonts/blender tools/benchmarks.
- No model weights, test dataset images, large media, checkout histories or nested \`.git\` directories are committed.
- All upstream copyright notices and original licenses remain in their vendor directories. Review their licenses separately before redistribution, commercial use, or deployment.

Regenerate vendored code from pinned hashes:
\`\`\`bash
python3 scripts/vendor_sources.py
python3 -m unittest discover -s tests -v
\`\`\`

Updating \`configs/upstreams.json\` requires re-running the CPU checks and actual CUDA installation on a GPU host. Vendored files are not handwritten and should normally not be edited in place: patch wrappers or document a vendor patch.

This project uses three isolated Conda environments, reflecting incompatible upstream PyTorch/CUDA pins. The initial CPU validations only prove basic wiring and source syntax. They do **not** prove CUDA compilation, convergence, dataset metrics, or mesh/UV fidelity.
