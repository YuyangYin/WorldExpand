# NVSPano-Project unit tests (TDD, pure-logic / param-contract layer).
#
# These are stdlib-`unittest` tests that run WITHOUT GPU, model weights, the v2
# dataset, or the conda env -- so they run in seconds in the editing environment
# itself. Scope is the deterministic glue/contract layer only (param derivation,
# meta.json train<->infer contract, path/uid construction, ...). GPU/data/quality
# checks stay in the smoke suite (docs/smoke_test.md) + eval metrics.
#
# Run:  python3 -m unittest discover -s tests -v
