# Data and third-party materials

The Unity Tennis environment binary is excluded. Download the [official Udacity Linux headless environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) separately under its applicable terms. The included `vendor/` directory is the legacy Unity ML-Agents client, with its original copyright and Apache 2.0 notices preserved. The official Udacity reinforcement-learning repository is MIT licensed; see `third_party/udacity-drl-MIT.txt`. Those licenses do not confer rights to the Unity environment binary.

The new Mac reference acquires the [official Intel Mac Tennis environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip). Its executable SHA-256 is `6811730d75a8130daa145ca21dd0a2c0f4995f7dcd55f1f891c6318629adfa4b`. Apple Rosetta runs this Intel simulator on the M2 Max. The simulator binary remains excluded and subject to its original terms; the scoped MIT tooling license does not grant binary rights. Fresh raw rollout records, source hashes and selected policy hashes are in `results/fresh-v1/`.


## Equal-interaction v2 evidence

`results/robust-v2/` retains ten fresh local trainings, their exact source/settings, all ten selected actor state dictionaries, training/validation histories and 2,000 final episodes including random/zero controls. The five training seeds per method are independent new trainings; nested evaluation episodes are not additional training replications. The official simulator is acquired separately and not redistributed. Source archives preserve the existing Unity/ML-Agents and Udacity notices. The new TD3 implementation follows published equations with attribution and claims no new algorithm.
