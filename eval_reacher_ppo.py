import os

from env.reacher import ReacherEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

import matplotlib.pyplot as plt

# Code adapted from stable_baselines

def main():
    model = PPO2.load("./data/reacher2.zip")     
    # Enjoy trained agent
    env = ReacherEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)


if __name__ == '__main__':
    main()
