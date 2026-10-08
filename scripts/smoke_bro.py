import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from DoublePendulum.nn import bro
print('Imported!')

learner = bro.BRO_Learner(dt=10.0, buffer_length=1e4)
print('Initialized!')

learner.train(progress=True, num_episodes=5)

learner.dump()
