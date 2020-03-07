import os

#from gym_wrapper import URRobotGym
from env.reacher_wall import ReacherWallEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

import matplotlib.pyplot as plt

# Code borrowed from stable_baselines

def main():
    # create the environment
    log_dir = "tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = ReacherWallEnv(render=False)
    env = Monitor(env, log_dir, allow_early_resets=True)
    model = PPO2('MlpPolicy', env, verbose=1).learn(50000)
    model.save('./ppo2.zip')

    # plot training progress
    results_plotter.plot_results([log_dir], 50000, results_plotter.X_TIMESTEPS, "PPO UR Robot")
    plt.show()


    del model # remove to demonstrate saving and loading

    model = PPO2.load("./ppo2.zip")     
    # Enjoy trained agent
    env = ReacherWallEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()

    #model = PPO2(MlpPolicy, env, verbose=1)
    #model.learn(total_timesteps=25000)
    #model.save("Robot")

if __name__ == '__main__':
    main()
