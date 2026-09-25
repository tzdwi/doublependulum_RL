import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DoublePendulum.nn import sac
print('Imported!')

learner = sac.SAC_Learner(dt=10.0)
print('Initialized!')

learner.train(progress=True)

learner.dump()
