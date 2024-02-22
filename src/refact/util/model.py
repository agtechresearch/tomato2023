import torch
from torch import nn
from torch.autograd import Variable

import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np

# RNN 모델 정의
class RNN(nn.Module):
    name = "RNN"
    def __init__(self, input_dim=1, hidden_size=32, output_dim=1,
                 device="cpu", taskType="regression"):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    name = "LSTM"
    def __init__(self, input_dim=1, output_dim=1,
                  hidden_size=64, num_layers=1,
                  device="cpu", taskType="regression"):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)   
        # self._task = {
        #     "regression": lambda x: x,
        #     "classification": lambda x: (self.sigmoid(x) > 0.5).float(),
        #     "multi-class classification": lambda x: torch.argmax(self.softmax(x), dim=1),
        # }

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)
        
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        # out = self._task[self.taskType](out)
        # out.requires_grad_(True)
        
        return out
    

class DoubleLayerLSTM(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                  hidden_size=64, num_layers=1,
                  device="cpu"):
        super(DoubleLayerLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)


    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        ).to(self.device)

        lstm_out1, _ = self.lstm1(x, (h_0, c_0))
        lstm_out2, _ = self.lstm2(lstm_out1)
        h_out = lstm_out2[:, -1, :].view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out

    

# Positional Encoding for Transformer
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(_PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * \
                (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
# Model definition using Transformer
class Transformer(nn.Module):
    name = "Transformer"
    def __init__(self, input_dim=1, output_dim=1, 
                 d_model=64, nhead=4, num_layers=1,
                 device="cpu"):
        super(Transformer, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x
    

####################
    

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
    def __init__(self, capacity, device="cpu"):
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)  # GPU로 이동
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)  # GPU로 이동
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)  # GPU로 이동
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)  # GPU로 이동
        dones = torch.BoolTensor(np.array([e.done for e in batch])).to(self.device)  # GPU로 이동
        return states, actions, rewards, next_states, dones

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=int(1e5), 
                 batch_size=32, gamma=0.99, lr=1e-3, update_freq=4, 
                 target_update_freq=1000, device="cpu"):
        self.device = device
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
        self.replay_buffer = ReplayBuffer(buffer_size, device=device)
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # GPU로 이동
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