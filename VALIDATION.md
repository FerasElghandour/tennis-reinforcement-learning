# Release validation — October 17, 2026

- Python source syntax checked, including bundled legacy client files.
- Fresh `verify_agent.py` smoke test passed: a learning update, finite bounded actions, frozen target gradients and policy save/load equivalence.
- Published actor checkpoint loaded with `weights_only=True`; its actions have the expected shape and finite values in range.
- Training/evaluation CSV records rechecked: 875 training episodes; first qualifying full 100-episode window ends at episode 607; 100 evaluation episodes have mean 2.0670000308007004.
- Smoke/checkpoint checks ran on macOS, Python 3.14.3, PyTorch 2.14.1 and NumPy 2.5.4. These are additional compatibility checks, not the historical training environment.
- The Unity environment was not rerun or retrained for this release. Saved scores describe the historical Linux run and one seed.
