import os
import time
import subprocess
from BallCatch import BallCatch
from agent import Agent
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return image


def create_plot(episode_number, speeds, energy_transfer, total_reward):
    # not implemented yet

    return fig


def agent_train():
    game_name = "BallCatch"
    run_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_dir = os.path.join(os.path.join(
        os.path.expanduser('~')), 'Desktop/tensorboard_Data')

    port = 6006

    subprocess.Popen(
        f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)

    log_dir = log_dir + '/' + game_name + '_' + str(run_time)

    env = BallCatch()

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
                  gamma=gamma)

    # Set up TensorBoard output
    writer = SummaryWriter(log_dir=log_dir)

    num_episode = 1000

    speeds = [10, 8, 6, 4, 2, 1]
    energy_transfers = [1, 0.8, 0.6, 0.4, 0.2]

    episode_number = 0
    for speed in speeds:
        for energy_transfer in energy_transfers:
            for i_episode in range(num_episode):
                # in reset change the energy transfer and speed
                state, info = env.reset(
                    speed=speed, engergy_transfer_persentage=energy_transfer)
                done = False
                truncated = False
                total_length = 1
                total_reward = 0

                while not(done or truncated):
                    action = agent.act(state)
                    next_state, reward, done, truncated, info = env.step(
                        action)

                    total_reward += reward
                    total_length += 1

                    agent.remember(state, action, reward, next_state, done)

                    state = next_state
                    agent.replay()

                if done:
                    agent.decay_epsilon()

                writer.add_scalar("reward", total_reward, episode_number)
                writer.add_scalar("length", total_length, episode_number)
                writer.add_scalar(
                    "reward_rate", total_reward/total_length, episode_number)
                writer.add_scalar("epsilion", agent.epsilon, episode_number)

                writer.add_scalars(f'Speed_Energy', {
                    'speed': speed,
                    'energy_per': energy_transfer,
                    'reward': total_reward,
                }, episode_number)

                print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(
                    episode_number, total_reward, agent.epsilon, total_length))

                # Create the plots and log them to TensorBoard
                # fig1 = create_plot(episode_number, speed, energy_transfer, total_reward)

                # writer.add_image('Speed vs Total Reward', plot_to_image(fig1), 0)

                episode_number += 1

    env.close()
    writer.close()


agent_train()
