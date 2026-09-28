import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DoublePendulum.nn import sac
print('Imported!')

learner = sac.SAC_Learner(log_progress=True)
print('Initialized!')

learner.train(progress=True, make_plots=True, num_episodes=10000)

learner.dump()
