from util.log import plot_ts_result
from util.process import MyDataset, Modeling
from util.model import TransformerModel, LSTM

import numpy as np
import pandas as pd

df = pd.read_csv("data/merged_fill.csv") 
df["INNER_HMDT"] = (df["INNER_HMDT_1"] + df["INNER_HMDT_2"]) / 2
df["INNER_TPRT"] = (df["INNER_TPRT_1"] + df["INNER_TPRT_2"]) / 2

x_cols = ["EXTN_TPRT", "INNER_HMDT"] #"DWP_TPRT"를 넣으면 Transformer에서 거의 100점짜리 답안이 나옴
y_cols = ["INNER_TPRT"]
data = MyDataset(df, x_cols, y_cols)
train_loader, test_loader = data.preprocessing(train_ratio=0.8)

modeling = Modeling(model=TransformerModel, 
                    data=data, lr=0.001)

modeling.train(
    epochs=100,
    train_loader=train_loader,
    test_loader=test_loader,
)

true_val, pred_val = modeling.eval(train_loader, data.y_train)

batch_size = 32
X = df[x_cols][:len(train_loader)*batch_size]
y = df[y_cols][:len(train_loader)*batch_size]
X["pred"] = pred_val

# FIXED TICK SIZE

import torch
import gym
from gym import spaces
import numpy as np

class GreenhouseTSEnv(gym.Env):
    def __init__(self, data, tick=4): # data is a batch
        super(GreenhouseTSEnv, self).__init__()

        self.tick = tick
        self.history = data
        self.current_step = -1
        batch_size, input_size = data[0].shape
        self.max_steps = batch_size

        self.observation_space = spaces.Box(
            low=-50, high=50, shape=(input_size,))
        # Action space: Up, Hold, Down
        self.action_space = spaces.Discrete(3)
        self.actions = {0: self.tick, 1: 0, 2: -self.tick}

    def reset(self):
        self.current_step = -1
        return self.history[0][self.current_step]

    def step(self, action):
        self.current_step += 1
        pred = self.history[0][self.current_step][-1]
        answer = self.history[1][self.current_step][-1]
        
        # if abs(pred-answer) <= self.tick:
        #     action = 1
        # pred += self.actions[action]
        pred += self.actions[action] \
            if abs(pred-answer) >= self.tick else \
            self.actions[action]/2
        
        reward = - abs(torch.tensor(pred) - answer)
        observation = self.history[0][self.current_step]
        done = self.current_step > self.max_steps 

        return observation, reward, done, {"new_pred": pred}
    
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network for Q-function approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(device)  # GPU로 이동
        actions = torch.LongTensor([e.action for e in batch]).to(device)  # GPU로 이동
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device)  # GPU로 이동
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)  # GPU로 이동
        dones = torch.BoolTensor([e.done for e in batch]).to(device)  # GPU로 이동
        return states, actions, rewards, next_states, dones

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=int(1e5), batch_size=32, gamma=0.99, lr=1e-3, update_freq=4, target_update_freq=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.q_network = DQN(state_size, action_size).to(device)  # GPU로 이동
        self.target_network = DQN(state_size, action_size).to(device)  # GPU로 이동
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # GPU로 이동
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())



window_size = 2000
for z in range(0, len(df), window_size):
    temp = X.iloc[z:z+window_size]
    env = GreenhouseTSEnv([
        temp.to_numpy(),
        y.iloc[z:z+window_size].to_numpy(),
    ])
    break

state_size = env.observation_space.shape[-1]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Define training parameters
epsilon_start = 1.0
epsilon_final = 0.001
epsilon_decay = 0.999
max_episodes = 10000
max_steps_per_episode = env.max_steps

epsilon = epsilon_start
for episode in range(1, max_episodes + 1):
    state = env.reset()
    episode_reward = 0
    for step in range(max_steps_per_episode):
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        episode_reward += reward
        if done:
            break
    epsilon = max(epsilon_final, epsilon * epsilon_decay)
    if episode % 50 == 0:
      print(f"Episode: {episode}, Reward: {episode_reward.item()}, Epsilon: {epsilon}")

torch.save(agent.q_network, 'q_model.pth')
torch.save(agent.target_network, 'target_model.pth')


from util.log import regression_results

new_pred_val = []
epsilon = 0.0  # 추론할 때는 무작위 탐험이 필요하지 않으므로 epsilon을 0으로 설정

state = env.reset()
episode_reward = 0

for step in range(max_steps_per_episode):
    pred = pred_val[step]
    action = agent.select_action(state, epsilon)
    next_state, reward, done, info = env.step(action)
    new_pred_val.append(info["new_pred"])  # 새로운 예측값을 저장
    
    state = next_state
    episode_reward += reward
    
    if done:
        break

plot_ts_result(true_val[:window_size], pred_val[:window_size], new_pred_val)

# 개선 전
regression_results(true_val[:window_size], pred_val[:window_size])
# 개선 후
regression_results(true_val[:window_size], new_pred_val)


true_val, pred_val = modeling.eval(test_loader, data.y_test)
batch_size = 32
X = df[x_cols][-len(test_loader)*batch_size:]
y = df[y_cols][-len(test_loader)*batch_size:]
X["pred"] = pred_val
# X["true"] = true_val

env = GreenhouseTSEnv([
    X.to_numpy(), y.to_numpy()
])
new_batch = X.shape[0]
max_steps_per_episode = env.max_steps


new_pred_val = []
epsilon = 0.0  # 추론할 때는 무작위 탐험이 필요하지 않으므로 epsilon을 0으로 설정

state = env.reset()
episode_reward = 0

for step in range(max_steps_per_episode):
    action = agent.select_action(state, epsilon)
    next_state, reward, done, info = env.step(action)
    new_pred_val.append(info["new_pred"])  # 새로운 예측값을 저장
    
    state = next_state
    episode_reward += reward
    
    if done:
        break
plot_ts_result(true_val[:new_batch], pred_val[:new_batch],
               new_pred_val, "test_result.png")

# 개선 전
regression_results(true_val[:new_batch], pred_val[:new_batch])
# 개선 후
regression_results(true_val[:new_batch], new_pred_val)