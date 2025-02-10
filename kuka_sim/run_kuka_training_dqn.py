import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim


from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet as p

from dqn import ReplayMemory, DQN, get_screen
from train import select_action, optimize_model


def main():
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20)
    env.cid = p.connect(p.DIRECT)

    plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    STACK_SIZE = 5
    TARGET_UPDATE = 1000
    LEARNING_RATE = 1e-4
    PATH = 'policy_dqn.pt'
    
    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    
    env.reset()
    
    # plt.figure()
    # plt.imshow(get_screen(env, device=device).cpu().squeeze(0)[-1].numpy(),cmap='Greys',
    #         interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()
    
    init_screen = get_screen(env, device=device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(STACK_SIZE, screen_height, screen_width, n_actions).to(device)
    target_net = DQN(STACK_SIZE, screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(10000, Transition)

    eps_threshold = 0
    
    # train
    num_episodes = 200
    # num_episodes = 10000000
    writer = SummaryWriter()
    total_rewards = []
    ten_rewards = 0
    best_mean_reward = None
    start_time = timeit.default_timer()
    
    for i_episode in range(num_episodes):
        print(f"Episode{i_episode}: ")
        # Initialize the environment and state
        env.reset()
        state = get_screen(env, device=device)
        stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
        for t in count():
            stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
            
            # Select and perform an action
            action = select_action(policy_net, stacked_states_t, n_actions, i_episode, eps_threshold, device=device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            print(f"action: {action}, reward: {reward}")

            # Observe new state
            next_state = get_screen(env, device=device)
            if not done:
                next_stacked_states = stacked_states
                next_stacked_states.append(next_state)
                next_stacked_states_t =  torch.cat(tuple(next_stacked_states),dim=1)
            else:
                next_stacked_states_t = None
                
            # Store the transition in memory
            memory.push(stacked_states_t, action, next_stacked_states_t, reward)

            # Move to the next state
            stacked_states = next_stacked_states
            
            # Perform one step of the optimization (on the target network)
            optimize_model(Transition, memory, policy_net, target_net, optimizer, device=device)
            if done:
                reward = reward.cpu().numpy().item()
                ten_rewards += reward
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-100:])*100
                writer.add_scalar("epsilon", eps_threshold, i_episode)
                if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
                    # For saving the model and possibly resuming training
                    torch.save({
                            'policy_net_state_dict': policy_net.state_dict(),
                            'target_net_state_dict': target_net.state_dict(),
                            'optimizer_policy_net_state_dict': optimizer.state_dict()
                            }, PATH)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                break
                
        if i_episode%10 == 0:
                writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode)
                ten_rewards = 0
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if i_episode>=200 and mean_reward>50:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode+1, mean_reward))
            break


    print('Average Score: {:.2f}'.format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    env.close()


if __name__ == "__main__": 
    main()