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
        self.model_based = model_based

        # currently working on the model part (not yet implemented)

        self.batch_size = batch_size 
        self.memory_size = memory_size 
        self.gamma = gamma 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay_rate = 0.9995 

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


        if model_based:
            """ model is esitimation of the env, 
            simple model just calculates estimate s(t+1), r(t+1) output if given input of s(t), a(t) """
            self.model_network = Model_Network_vanilla(self.state_size+1, self.state_size+1, self.hidden_size).to(self.device)
            self.model_gamma = gamma 
            self.model_epsilon = 1.0 
            self.model_epsilon_min = 0.01 
            self.model_epsilon_decay_rate = 0.9995
            self.model_optimizer = optim.Adam(self.model_network.parameters() , lr=self.learning_rate)
            self.model_criterion = nn.MSELoss()  
            self.model_train_n = 0 
            self.model_train_min_for_simul = 100
            self.model_max_simulation_n = 50 
            self.model_loss = 100 
            self.model_loss_bound = 0.1 



    def act(self, state, max_q_action= True, model_acting=False): 
        if self.model_based and np.random.rand() <= 0.05 and not(model_acting):
            self.model_simulate(state, 10)
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
    def model_simulation_remember(self, state, action, reward, next_state, done):
        self.model_experience_memory.append((state, action, reward, next_state, done))

        

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
        if self.model_based:
            self.model_train(states, actions, next_states, rewards)


    def model_train(self, states, actions, next_states, rewards):
        self.model_train_n += 1 
        state_action_pair = torch.cat((states, actions.unsqueeze(1)), dim=1)
        reward_next_state_pair = torch.cat((next_states, rewards.unsqueeze(1)), dim=1)
        model_next_states, model_next_reward = self.model_network(state_action_pair)

        current_model_output = torch.cat((model_next_states, model_next_reward), dim=1)
        target_model_output = reward_next_state_pair
        loss = self.model_criterion(current_model_output, target_model_output)
        self.model_loss = loss.item()
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def model_simulate(self, state, simulation_size): 
        self.model_experience_memory = deque(maxlen=self.memory_size)
        

        if (self.model_loss > self.model_loss_bound) or (self.model_train_n < self.model_train_min_for_simul):
            return 
        
        for simulation_num in range(simulation_size):
            done = False 
            simul_n = 0

            while not(done):
                simul_n +=1 
                if simul_n >= self.model_max_simulation_n:
                    done = True 

                action = self.act(state, model_acting=True)

                action = np.array([action])
            
                state_action_pair = np.concatenate((state, action))
                state_action_tensor = torch.FloatTensor(state_action_pair).to(self.device)
                next_state, reward = self.model_network(state_action_tensor)

                next_state = next_state.cpu().detach().numpy()
                next_state = np.round(next_state)
                reward = float(reward.cpu().detach().numpy())
          

                if reward > 5:
                    done = True 
                self.model_simulation_remember(state, action, reward, next_state, done)
                state = next_state

        self.simulation_learn() 

    def simulation_learn(self):
        if len(self.model_experience_memory) < self.batch_size:
            return
        # middle_batch_size = 4 * self.batch_size
        # if len(self.model_experience_memory) >= middle_batch_size:
        #     model_sub_memory = random.sample(self.model_experience_memory, middle_batch_size)
        # else:
        #     model_sub_memory = self.model_experience_memory
        # TD_PER = self.make_TD_PER(model_sub_memory)
        # minibatch_idx = np.random.choice(list(range(len(model_sub_memory))), self.batch_size, p=TD_PER)
        # minibatch = [model_sub_memory[i] for i in minibatch_idx]
        minibatch = random.sample(self.model_experience_memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        ## greedly optimized with TD error 
        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions)
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate  * self.epsilon)
       
    
