from smac.env import StarCraft2Env
import numpy as np

TASKS = ["3m", "8m", "25m", "2s3z", "3s5z", "MMM", "5m_vs_6m",
    "8m_vs_9m", "10m_vs_11m","27m_vs_30m", "3s5z_vs_3s6z",
    "MMM2", "2m_vs_1z", "2s_vs_1sc", "3s_vs_3z", "3s_vs_4z",
    "3s_vs_5z", "6h_vs_8z", "corridor", "bane_vs_bane", 
    "2c_vs_64zg", "1c3s5z"]


def main():
    for task in TASKS:
        env = StarCraft2Env(map_name=task)
        env_info = env.get_env_info()
        print ("%s => %d" % (task, env_info['episode_limit']))
        env.close()

if __name__ == "__main__":
    main()
