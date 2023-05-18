import os
import time 
import subprocess 
from BallCatch import BallCatch
from agent import Agent
from torch.utils.tensorboard import SummaryWriter 


def agent_train(model_based = False):
    game_name = "BallCatch"
    run_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/tensorboard_Data')
    print("model based is ", model_based)

    port = 6006 

    subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)

    log_dir = log_dir + '/' + game_name + '_' + str(run_time)

    env = BallCatch(obs_frame=1)

    state_size = env.state_n
    action_size = env.action_n

    hidden_size = 256
    learning_rate = 0.001 
    memory_size = 10000 
    batch_size = 64
    gamma = 0.99 

    agent = Agent(state_size=state_size, action_size=action_size,
                  hidden_size=hidden_size, learning_rate=learning_rate,
                  memory_size=memory_size, batch_size=batch_size,
                  gamma=gamma, 
                  model_based=model_based)
    print(model_based)
    

    # Set up TensorBoard output
    writer = SummaryWriter(log_dir=log_dir)


    num_episode = 5000

    for i_episode in range(num_episode):
        state, info = env.reset()
        done = False
        truncated = False 
        total_length = 1 
        total_reward = 0 

        while not(done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            total_reward += reward 
            total_length +=1 

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            agent.replay()

        
        if done:
            agent.decay_epsilon() 


        writer.add_scalar("reward", total_reward, i_episode) 
        writer.add_scalar("length", total_length, i_episode)
        writer.add_scalar("reward_rate", total_reward/total_length, i_episode)
        writer.add_scalar("epsilion", agent.epsilon, i_episode)

        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(i_episode, total_reward, agent.epsilon, total_length))

    env.close()
    writer.close()  


agent_train(model_based = True) 