import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if num_envs < 1:
        raise SystemExit("environment count must be at least 1")

    from DoublePendulum.nn import ddpg
    print("Imported!")

    learner = ddpg.DDPG_Learner(buffer_length=1e4, num_envs=num_envs)
    print(f"Initialized, environments {num_envs}")

    learner.train(progress=True, num_episodes=5)
    learner.dump()


if __name__ == "__main__":
    main()
