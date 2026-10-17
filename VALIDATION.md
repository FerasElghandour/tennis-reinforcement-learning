# Methods and validation

The 28-test suite covers score identities, truncation-aware windows, block resampling, replay ownership, independent random generators, invalid transitions, mode-preserving inference and exported-policy parity. New numerical oracles check hand-computed DDPG targets, terminal masking, actor gradients, Polyak updates and BatchNorm buffers. Legacy-client checks protect per-instance pipes, bounded waits and complete shutdown. Exploration fixtures check seeded warmup and preservation of the training budget during evaluation.

```sh
python -m pip install -e ".[agent,dev]"
python -m pytest -q
python verify_fresh_results.py --results results/fresh-v1
```

Fixtures isolate correctness; they are not performance benchmarks. The [fresh campaign](results/fresh-v1/RESULTS.md) separately reports six complete training runs, validation-only selection, held-out policy/baseline comparisons and all raw records. Its offline audit checks scores, counts, checkpoint selection and actor fingerprints. The historical saved actor's 300 new episodes evaluate one policy, not three training seeds.

The original interactive report remains generated from the original historical data and records implementation/input hashes in `docs/metrics.json`. It is a historical view, while the new campaign has its own result page. Browser checks of that existing report cover desktop/mobile tables and downloads.

The fresh Mac Unity reference uses Python 3.11.15, Torch 2.5.1, NumPy 1.26.3, protobuf 3.20.3, gRPC 1.84.0 and the official Intel simulator under Rosetta. Exact source archives and dependency snapshots accompany the results. An earlier native transport failure is retained explicitly. The critic uses local state/action observations, not centralized joint observations. Simulation results do not establish real-world control safety or cross-platform determinism.
