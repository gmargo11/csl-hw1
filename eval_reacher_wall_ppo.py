import os

from env.reacher_wall import ReacherWallEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

import matplotlib.pyplot as plt

# Code adapted from stable_baselines

def main():
    #model = PPO2.load("./data/reacher_wall_better_reward1.zip")     
    model = PPO2.load("./data/reacher_wall1.zip")   
    # Enjoy trained agent
    env = ReacherWallEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)


if __name__ == '__main__':
    main()
