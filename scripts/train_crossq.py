import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if num_envs < 1:
        raise SystemExit("environment count must be at least 1")

    from DoublePendulum.nn import crossQ
    print("Imported!")

    learner = crossQ.CrossQ_Learner(log_progress=True, num_envs=num_envs)
    print(f"Initialized, environments {num_envs}")

    try:
        learner.load()
        print("Successfully loaded previous weights")
    except Exception:
        print("Initializing with fresh weights")

    learner.train(progress=True, make_plots=True, num_episodes=100000)
    learner.dump()


if __name__ == "__main__":
    main()
