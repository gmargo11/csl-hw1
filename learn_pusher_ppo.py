import os

from env.pusher import PusherEnv
from VecMonitor import VecMonitor

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

import matplotlib.pyplot as plt
import numpy as np

# Code adapted from stable_baselines

def main(): 
    # create the environment

    num_trials = 3
    trial_length = 500000

    results = []

    for i in range(num_trials):
        log_dir = "tmp/gym/pusher/%d"%(i)
        os.makedirs(log_dir, exist_ok=True)
        #env = PusherEnv(render=True)
        #env = Monitor(env, log_dir, allow_early_resets=True)
        env = make_vec_env(PusherEnv, n_envs=10)
        env = VecMonitor(env)

        model = PPO2('MlpPolicy', env, verbose=1, seed=i, cliprange=0.1).learn(trial_length)
        model.save('./data/pusher%d.zip'%(i))

        results += [env.get_episode_rewards()]


    # plot training result

    results = np.array(results)
    print(results.shape)
    mean_reward = np.mean(results, 0)
    std_reward = np.std(results, 0)

    smoothing_window = 5
    mean_reward = np.convolve(mean_reward, np.ones((smoothing_window,))/smoothing_window, mode='valid')
    std_reward = np.convolve(std_reward, np.ones((smoothing_window,))/smoothing_window, mode='valid')
    t = np.array(range(1, mean_reward.shape[0]+1))

    plt.figure()
    plt.plot(t, mean_reward, color='blue')
    plt.fill_between(t, mean_reward+std_reward, mean_reward-std_reward, facecolor='blue', alpha=0.5)
    plt.title("Pusher Reward during PPO Training")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.show()


    del model

    model = PPO2.load("./data/pusher1.zip")
    # Enjoy trained agent
    env = PusherEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)


if __name__ == '__main__':
    main()
