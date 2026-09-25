import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DoublePendulum.nn import ddpg
print('Imported!')

learner = ddpg.DDPG_Learner()
print('Initialized!')

learner.train(progress=True, make_plots=True)

learner.dump()
