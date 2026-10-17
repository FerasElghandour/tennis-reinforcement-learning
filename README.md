# Tennis with Shared-Policy DDPG

Train two agents in the Udacity Tennis environment using one shared actor, critic and replay buffer.

## Completion

Udacity project: **Collaboration and Competition**. Passed on **2026-10-06**, verified against the authenticated Udacity project-pass email. The receipt is kept privately.

This is a cleaned portfolio edition of the recovered implementation associated with that project. Udacity's email confirms the project passed; it does not certify this Git commit or identify the accepted archive by hash.

## Approach

Each agent uses its local 24-dimensional stacked observation and two continuous actions. Experience is pooled for DDPG updates. Despite the legacy class name `MADDPGAgent`, this implementation has no centralized critic and is not centralized-critic MADDPG.

## Recorded results

Saved training records contain 875 episodes. The first qualifying 100-episode window ends at episode 607 with mean score 0.510000. A separate 100-episode evaluation of the saved policy, without exploration or learning, has mean score 2.067000. These are records from one seed, not a multi-seed benchmark.

These figures come from saved execution records. Release validation is described in `VALIDATION.md`.

## Run

Use Linux and Python 3.11 for the historical environment setup.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_agent.py
```

Download and extract the official environment separately, then:

```sh
python Tennis.py --mode probe --env Tennis_Linux_NoVis/Tennis.x86_64
cp -R run-seed-0 run-local-evaluation
python Tennis.py --mode evaluate --env Tennis_Linux_NoVis/Tennis.x86_64 --out run-local-evaluation --eval-episodes 100
```

The copied output folder preserves the published evaluation records. To train a new policy, use `--mode train --out run-new-training`. See `Tennis.py --help` for the remaining arguments. Saved actor/critic weights and the score CSVs are included in `run-seed-0/`.

See [DATA.md](DATA.md) for inputs, attribution and excluded materials. Source, saved score records and model checkpoints are included; the environment binary is obtained separately.

## Limitations

The legacy Unity v0.4 client and supplied environment binary are required. The historical run used the Udacity Linux workspace, Python 3.11.0rc1 and CPU PyTorch 2.5.1. Other platforms have not been validated with the environment. The optional 0.75 training target was not reached.

## Provenance and rights

Prepared by Mohamed Feras Elghandour with AI assistance in implementation, debugging, analysis and documentation. This repository does not claim unaided coding or production deployment. Udacity supplied the educational context; third-party tools and inputs retain their own rights. Course lesson text, assessment prompts, rubrics, answer keys, pass-email bodies and authenticated classroom captures are excluded.

No new repository-wide software license is granted in this release. Existing third-party license notices remain applicable; see [DATA.md](DATA.md) and any `third_party/` or `vendor/` notices. This is an educational portfolio project, without a claim of Udacity endorsement.
