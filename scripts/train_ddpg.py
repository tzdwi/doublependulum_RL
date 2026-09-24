import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DoublePendulum.nn import ddpg
print('Imported!')

learner = ddpg.DDPG_Learner(dt=10.0)
print('Initialized!')

learner.train()
