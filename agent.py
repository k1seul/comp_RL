from network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 

class Agent():
    """
    """
    def __init__(self, state_size, action_size, hidden_size, learning_rate,
                 memory_size, batch_size, gamma, policy_network= "Q_network", 
                 model_based=False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device))


        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # currently working on the model part (not yet implemented)

        self.batch_size = batch_size 
        self.memory_size = memory_size 
        self.gamma = gamma 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay_rate = 0.99995 

        # exeprience memory for the batch learning (batch is randomly sampled from the memory)
        self.experience_memory = deque(maxlen=self.memory_size)


        if policy_network == 'Q_network':
            print("policy network is currently q network")
            self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        elif policy_network == 'LSTM_Q':
            print("policy_network is currently LSTM network")
            self.q_network = LSTM_Q(state_size, action_size, hidden_size).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()



    def act(self, state, max_q_action= True): 
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size) 
        else:
            state_tensor = torch.Tensor(state).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            if max_q_action:
                return np.argmax(q_values.cpu().numpy())
            else: 
                raise NotImplementedError("currently working on it!!")
    
    def remember(self, state, action, reward, next_state, done):
        self.experience_memory.append((state, action, reward, next_state, done))
        

    def replay(self, uniformed_sample = True, TD_sample = False): 
        if len(self.experience_memory) < self.batch_size:
            return 
        else:
            minibatch = random.sample(self.experience_memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        ## greedly optimized with TD error 
        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1))
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate  * self.epsilon)
       
    
