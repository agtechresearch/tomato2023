#%%
from util.log import plot_ts_result, regression_results
from util.model import DQNAgent

import torch
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

df = pd.read_csv("data/merged_fill.csv") 
df["INNER_HMDT"] = (df["INNER_HMDT_1"] + df["INNER_HMDT_2"]) / 2
df["INNER_TPRT"] = (df["INNER_TPRT_1"] + df["INNER_TPRT_2"]) / 2

x_cols = ["EXTN_TPRT", "INNER_HMDT"] 
y_cols = ["INNER_TPRT"]
pred_df_train = pd.read_csv(f"data/pred_{y_cols[0]}_train.csv")
pred_df_test = pd.read_csv(f"data/pred_{y_cols[0]}_test.csv")

class GreenhouseTSEnv(gym.Env):
    def __init__(self, data, tick=3): # data is a batch
        super(GreenhouseTSEnv, self).__init__()

        self.tick = tick
        self.history = data
        self.current_step = 0
        self.nums_data, input_size = data[0].shape
        self.max_steps = self.nums_data
        self.punish = 2

        self.observation_space = spaces.Box(
            low=-50, high=50, shape=(input_size,))
        self.action_space = spaces.Discrete(3)
        self.actions = {0: self.tick, 1: 0, 2: -self.tick}

    def reset(self):
        self.current_step = 0
        return self.history[0][self.current_step]

    def step(self, action):
        pred = self.history[0][self.current_step][-1]
        answer = self.history[1][self.current_step]
        before_diff = abs(torch.tensor(pred) - answer)
        pred += self.actions[action]
        
        after_diff = abs(torch.tensor(pred) - answer)
        if before_diff < after_diff:
            after_diff *= self.punish
        reward = - after_diff/self.nums_data

        self.current_step += 1
        done = self.current_step >= self.max_steps 
        observation =  None if done else self.history[0][self.current_step]
        

        return observation, reward, done, {"new_pred": pred}

    def step_infer(self, action):
        pred = self.history[0][self.current_step][-1]
        pred += self.actions[action]
        
        self.current_step += 1
        done = self.current_step >= self.max_steps 
        observation =  None if done else self.history[0][self.current_step]
        

        return observation, None, done, {"new_pred": pred}

# Env가 잘 만들어졌는데 실행시키는 코드
batch_size = 4
X = df[x_cols][:100]
X["pred"]  = pred_df_train.iloc[:,-1].values[:100]
X.reset_index(drop=True, inplace=True)
y = pred_df_train["answer"].reset_index(drop=True)[:100]

env = GreenhouseTSEnv([
    X.iloc[:batch_size].to_numpy(),
    y.iloc[:batch_size].to_numpy()
])

observation = env.reset()
for i in range(env.max_steps):
    action = env.action_space.sample()  # Random action for now
    observation, reward, done, _ = env.step(action)
    print(action, observation, reward, done)
    if done:
        print("done")
        break

@dataclass
class Epsilon:
    current: float = 1.0 # start value
    final: float = 0.2
    decay: float = 0.995

# MOVING_AVERAGE_WINDOW = 10
def log_train(episode, episode_rewards, current_epsilon, fname=""):
    print(f"Episode: {episode}, Reward: {episode_rewards[-1].item()}, Epsilon: {current_epsilon}")
    plt.plot(episode_rewards)
    # moving_average = np.convolve(
    #     episode_rewards, np.ones(MOVING_AVERAGE_WINDOW)/MOVING_AVERAGE_WINDOW, mode='valid')
    # plt.plot(np.arange(MOVING_AVERAGE_WINDOW-1, len(episode_rewards)), 
    #         moving_average, color='red')
    plt.savefig(f"rewards_{fname}.png")

def prepare_data(model_name, pred_df, train_mode=True, window_size=None):
    global df, x_cols, y_cols
    if train_mode:
        X = df[x_cols].iloc[:pred_df.shape[0]]
    else:
        X = df[x_cols][-pred_df.shape[0]:]
    X["pred"] = pred_df[model_name].values
    X.reset_index(drop=True, inplace=True)
    y = pred_df["answer"].reset_index(drop=True)

    if window_size is None:
        window_size = X.shape[0]
        print(X.shape)

    return (X.iloc[:window_size],
            y.iloc[:window_size])
        
def prepare_model():
    state_size = env.observation_space.shape[-1]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size, action_size, device=device)
    return agent

# def main_infer(model_name, pred_df_test, env, agent, ploting=True):
#     global df, x_cols, y_cols

#     X, y = prepare_data(model_name, pred_df_test, train_mode=False)
#     env = GreenhouseTSEnv([X.to_numpy(), y.to_numpy()], tick=3)

#     new_pred_val = [] 
#     epsilon = 0.0  # 추론할 때는 무작위 탐험이 필요하지 않으므로 epsilon을 0으로 설정
#     state = env.reset()

#     for step in range(env.max_steps):
#         action = agent.select_action(state, epsilon)
#         next_state, _, done, info = env.step_infer(action)
#         new_pred_val.append(info["new_pred"])  # 새로운 예측값을 저장
#         state = next_state
        
#         if done:
#             break
#     result = regression_results(y, new_pred_val)
#     if ploting:
#         plot_ts_result(y, X["pred"], new_pred_val)
#         plot_ts_result(new_pred_val, X["pred"])
#     return result
    


def main_train(env, agent, episode_rewards, epsilons, max_episodes, model_name, pred_df_test, tick=4):

    test_X, test_y = prepare_data(model_name, pred_df_test, train_mode=False)
    test_env = GreenhouseTSEnv([test_X.to_numpy(), test_y.to_numpy()], tick=tick)
    result = {
        "explained_variance": [],
        "r2": [],
        "mae": [],
        "mse": [],
        "rmse": [],
    }

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for _ in range(env.max_steps): # max_steps_per_episode
            action = agent.select_action(state, epsilons.current)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            
        epsilons.current = max(epsilons.final, epsilons.current * epsilons.decay)
        episode_rewards.append(episode_reward)
        if episode % 25 == 0:
            log_train(episode, episode_rewards, epsilons.current)
            # y = pred_df["answer"].reset_index(drop=True)
            # main_infer(model_name, pred_df_test, env, agent, ploting=False)
        new_pred_val = [] 
        state = test_env.reset()
        for _ in range(test_env.max_steps):
            action = agent.select_action(state, 0)
            next_state, _, done, info = test_env.step_infer(action)
            new_pred_val.append(info["new_pred"])  # 새로운 예측값을 저장
            state = next_state
            
            if done:
                break
        for k, v in regression_results(test_y, new_pred_val, print_res=False).items():
            result[k].append(v)
    return result

#%%
model_name = "RNN"
X, y = prepare_data(model_name, pred_df_train, window_size=2000)
env = GreenhouseTSEnv([X.to_numpy(), y.to_numpy()], tick=4)
agent = prepare_model()
epsilons = Epsilon()
episode_rewards = []
max_episodes = 350
result = main_train(env, agent, episode_rewards, epsilons, max_episodes, model_name, pred_df_test)