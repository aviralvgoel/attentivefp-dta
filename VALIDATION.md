# Validation of this release

Validated on 2 November 2026 in an isolated CPU environment. The original demo,
the roadmap document and the user's server data were not edited.

## Completed checks

- 59 pytest tests passed, with real RDKit/PyG/AttentiveFP model execution.
- PCBA synthetic pretraining, scratch DTA, frozen transfer, staged fine-tuning,
  optional atom-residue attention and explicit test export completed.
- Interrupted/resumed PCBA and DTA runs matched uninterrupted model weights
  bit-for-bit on the tested CPU fixture.
- A tiny locally initialized ESM model tested actual tokenizer/model feature
  extraction, special-token removal, chunking and mean/residue cache output.
- Label masks, zero-bond graphs, graph batching, stable molecule/pair IDs,
  schema rejection, stale protein cache rejection, split/entity leakage,
  frozen-encoder dropout/gradient behavior and safe output handling were tested.
- Concordance index matched brute-force pair comparisons across ten tied-label
  and tied-prediction fixtures.
- Full PCBA feature re-verification and pretraining exposure audit passed on
  the synthetic fixture.
- Python compilation, Ruff static checks and formatting, and Bash syntax checks
  passed.

## Not established here

- Execution on the user's NVIDIA RTX PRO 4500 Blackwell or its driver stack.
- Full PCBA/KIBA/DAVIS training, data quality, convergence or predictive accuracy.
- Download/access to research datasets or pretrained ESM-2 weights on the server.
- Scientific superiority of transfer learning or attention over the baselines.
- Bitwise determinism across GPU architectures, package versions or worker counts.

Run `python doctor.py --cuda` and `python smoke.py --device cuda` on the server
before full experiments. The smoke labels/features are synthetic and are not
research results. Undefined metrics are represented as JSON null rather than
invalid NaN tokens. Test metrics are never computed by the fit scripts.

## Key tested packages

| Package | Version observed in the test environment |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.9.1+cpu |
| PyTorch Geometric | 2.8.1 |
| Transformers | 4.57.6 |
| RDKit | 2026.9.1 |
| NumPy | 2.5.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| pytest | 9.2.0 |

These are recorded observations, not a recommendation to replace your working
GPU environment with a CPU build. Every real run also saves its own environment
metadata. `pyproject.toml` contains supported dependency ranges; the CUDA build
must be selected independently for the actual GPU and driver.
