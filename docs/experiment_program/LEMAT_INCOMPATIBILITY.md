# LeMat-GenBench Compatibility Record

Status: registered before model-result inspection

The active environment is Python 3.10 with PyTorch 2.4.0+cu121. The current
official LeMat-GenBench environment requires Python >=3.11 and PyTorch >=2.6,
and its preferred UMA/Orb evaluator assets are not present in the registered
model directory. Replacing the mandated environment would invalidate the
four-week lock and risks changing the generator stack.

Consequences:

1. this cycle does not claim an official LeMat leaderboard result;
2. no direct numerical rank against official LeMat submissions is allowed;
3. the project reports analogous attempt-level validity, stability, novelty,
   symmetry, failure, and efficiency fields under its frozen CHGNet,
   MatterSim, and MACE evaluator panel;
4. any future exact LeMat run must use a separately registered environment and
   cannot be retroactively mixed into this protocol.
