Legacy Unity ML-Agents client initially copied from the Udacity project workspace, unityagents and communicator_objects directories. Original project: https://github.com/Unity-Technologies/ml-agents . Copyright Unity Technologies; Apache 2.0. This client implements the v0.4 API required by the supplied Tennis environment. Local modifications are identified below.
# Local compatibility correction, 17 October 2026

Feras Elghandour changed `unityagents/rpc_communicator.py` so each servicer instance owns its own multiprocessing pipe. The original class-level pipe was shared across environment instances; closing a probe or validation environment closed another communicator's handle. Original Apache 2.0 licensing and notices remain in effect. This modification is distributed under those upstream terms, separately from the root file-scoped MIT grant.

The same file now bounds initialization acknowledgements and step responses to 30 seconds, waits for server shutdown, and closes both pipe endpoints. These are transport lifecycle changes; model and reward definitions are unchanged. The Mac reference uses gRPC 1.84.0 after an observed native shutdown crash with 1.58.0. The failed attempt is retained separately from completed campaigns.

