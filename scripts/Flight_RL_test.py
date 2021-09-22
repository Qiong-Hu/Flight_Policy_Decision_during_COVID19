#!/usr/bin/env python
# coding: utf-8

# In[141]:


import gym
import numpy as np
import copy
import matplotlib.pyplot as plt


# # Functions to show results

# In[554]:


def plot_states(states, title):
    fig, ax = plt.subplots()
    l1, = ax.plot(np.array(states)[:, 0], 'black')
    l2, = ax.plot(np.array(states)[:, 1], 'coral')
    l3, = ax.plot(np.array(states)[:, 2], 'red')
    l4, = ax.plot(np.array(states)[:, 3], 'green')
    ax.set_xticks(np.arange(0,160,10))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_xlabel("Days")
    ax.set_ylabel("Population Ratio")
    ax.set_title("Local city, " + title)
    ax.legend((l1, l2, l3, l4), ("Susceptible", "Exposed", "Infectious", "Recovered"))
    plt.show()

    fig, ax = plt.subplots()
    l1, = ax.plot(np.array(states)[:, 4], 'black')
    l2, = ax.plot(np.array(states)[:, 5], 'coral')
    l3, = ax.plot(np.array(states)[:, 6], 'red')
    l4, = ax.plot(np.array(states)[:, 7], 'green')
    ax.set_xticks(np.arange(0,160,10))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_xlabel("Days")
    ax.set_ylabel("Population Ratio")
    ax.set_title("Remote city, " + title)
    ax.legend((l1, l2, l3, l4), ("Susceptible", "Exposed", "Infectious", "Recovered"))
    plt.show()
    
    print(f"Local final (%): {(np.array(states)[-1, :4]*1e4).astype(np.int)/1e2}")
    print(f"Remote final (%): {(np.array(states)[-1, -4:]*1e4).astype(np.int)/1e2}")


# In[555]:


def plot_actions(actions, title):
    fig, ax = plt.subplots()
    l1, = ax.plot(np.array(actions)[:, 0], "blue")
    l2, = ax.plot(np.array(actions)[:, 1], "coral")

    ax.set_xticks(np.arange(0,160,10))
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_xlabel("Days")
    ax.set_ylabel("Percentage")
    ax.set_title("Flight decisions, " + title)
    ax.legend((l1, l2), ("Flight Number", "Seat Occupancy"))
    plt.show()
    
    print(f"Final flight num (%): {np.array(actions)[-1, 0]}")
    print(f"Final occupancy (%): {np.array(actions)[-1, 1]}")


# In[556]:


def plot_rewards(rewards, title):
    fig, ax = plt.subplots()
    ax.plot(rewards, "black")
    ax.set_xticks(np.arange(0,160,10))
    ax.set_xlabel("Days")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards, " + title)
    plt.show()


# In[557]:


def plot_ep_rewards(aggr_ep_rewards, title):
    fig, ax = plt.subplots()
    l1, = ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], 'blue')
    l2, = ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], 'green')
    l3, = ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], 'coral')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Episode Rewards, " + title)
    ax.legend((l3, l1, l2), ('max', 'avg', 'min'))
    plt.show()


# #  Pre-defined parameters

# In[558]:


remote_severity_assumptions = {
    "population": 1000000,          # e.g: about LA population
    "infectious": 50000             # e.g: LA case number
}

local_severity_assumptions = {
    "population": 1000000,          # e.g: assume identical inbound and outbound severity
    "infectious": 50000             # e.g: assum identical
}

flight_assumptions = {
    "flight_dis": 0.75,             # ratio, real_flight_hour/max_flight_hour (18.5h from Sigapore to New Jersey)
    "max_flight": 300,              # estimated max flight_num for the airport or area
    "full_capacity": 200            # estimated avg passenger_num per flight (may change due to airport or country)
}

virus_assumptions = {
    "delay_days": 14,               # num of days between infectious and detection
    "R0": 2.2,                      # reproductive number: suspectible -> exposed
    "latency_days": 5.2,            # avg incubation days: exposed -> infectious
    "infectious_days": 2.3          # avg recovery days: infectious -> recovered
}


# # Define env

# In[559]:


class FlightEnv(gym.Env):
    def __init__(self, remote_severity_params = remote_severity_assumptions, local_severity_params = local_severity_assumptions, flight_params = flight_assumptions, virus_params = virus_assumptions, time_delta = 1, max_steps = 150, flight_cost = 1, passenger_reward = 1, infectious_penalty = 10):
        super().__init__()
        
        self.remote_severity_params = remote_severity_params
        self.local_severity_params = local_severity_params
        self.flight_params = flight_params
        self.virus_params = virus_params

        self.time_delta = time_delta
        self.max_steps = max_steps
        self.flight_cost = flight_cost
        self.passenger_reward = passenger_reward
        self.infectious_penalty = infectious_penalty
        
        state = self.reset()
        
        self.observation_space = gym.spaces.Box(0, 1, (8, ))
        self.observation_space.n = 8
        self.action_space = gym.spaces.Box(0, 1, (2, ))
        self.action_space.n = 2
        

    
    def reset(self):
        self.local_susceptible = self.local_severity_params["population"] - self.local_severity_params["infectious"]
        self.local_exposed = 0
        self.local_infectious = self.local_severity_params["infectious"]
        self.local_recovered = 0

        self.remote_susceptible = self.remote_severity_params["population"] - self.remote_severity_params["infectious"]
        self.remote_exposed = 0
        self.remote_infectious = self.remote_severity_params["infectious"]
        self.remote_recovered = 0

        self.steps = 0
        self.reward = 0

        return self.calc_state()

    def calc_state(self):
        local_state = copy.deepcopy(np.array((self.local_susceptible, self.local_exposed, self.local_infectious, self.local_recovered)))
        local_state = local_state/self.local_severity_params["population"]
        remote_state = copy.deepcopy(np.array((self.remote_susceptible, self.remote_exposed, self.remote_infectious, self.remote_recovered)))
        remote_state = remote_state/self.remote_severity_params["population"]
        state = np.hstack((local_state, remote_state))
        return state

    # action: (flight_num, occupancy)
    # flight_num in [0, 1]; occupancy in [0, 1]
    # real flight num = flight_num * self.flight_params["max_flight"]
    # real passenger per flight = occupancy * self.flight_params["full_capacity"]
    def step(self, action):
        # Define all the paramaters in SEIR model
        spread_rate = self.virus_params["R0"]/self.virus_params["infectious_days"]      # beta in SEIR equation
        incubation_rate = 1/self.virus_params["latency_days"]  # sigma in SEIR equation
        recovery_rate = 1/self.virus_params["infectious_days"]  # gamma in SEIR equation

        # Save the prev state for updating
        prev_state = {}
        prev_state["local_susceptible"] = copy.deepcopy(self.local_susceptible)
        prev_state["local_exposed"] = copy.deepcopy(self.local_exposed)
        prev_state["local_infectious"] = copy.deepcopy(self.local_infectious)
        prev_state["local_recovered"] = copy.deepcopy(self.local_recovered)
        prev_state["remote_susceptible"] = copy.deepcopy(self.remote_susceptible)
        prev_state["remote_exposed"] = copy.deepcopy(self.remote_exposed)
        prev_state["remote_infectious"] = copy.deepcopy(self.remote_infectious)
        prev_state["remote_recovered"] = copy.deepcopy(self.remote_recovered)


        # Natural update local states after time_delta
        self.local_susceptible -= spread_rate * prev_state["local_susceptible"] * prev_state["local_infectious"] / self.local_severity_params["population"] * self.time_delta
        self.local_exposed += (spread_rate * prev_state["local_susceptible"] * prev_state["local_infectious"] / self.local_severity_params["population"] - incubation_rate * prev_state["local_exposed"]) * self.time_delta
        self.local_infectious += (incubation_rate * prev_state["local_exposed"] - recovery_rate * prev_state["local_infectious"]) * self.time_delta
        self.local_recovered += recovery_rate * prev_state["local_infectious"] * self.time_delta

        # Natural update remote states after time_delta
        self.remote_susceptible -= spread_rate * prev_state["remote_susceptible"] * prev_state["remote_infectious"] / self.remote_severity_params["population"] * self.time_delta
        self.remote_exposed += (spread_rate * prev_state["remote_susceptible"] * prev_state["remote_infectious"] / self.remote_severity_params["population"] - incubation_rate * prev_state["remote_exposed"]) * self.time_delta
        self.remote_infectious += (incubation_rate * prev_state["remote_exposed"] - recovery_rate * prev_state["remote_infectious"]) * self.time_delta
        self.remote_recovered += recovery_rate * prev_state["remote_infectious"] * self.time_delta


        # Update influence of flight
        passenger_num = action[0] * action[1] * self.flight_params["max_flight"] * self.flight_params["full_capacity"]

        # Inbound flight influence for local
        self.local_susceptible *= 1 - passenger_num/self.local_severity_params["population"]
        self.local_exposed *= 1 - passenger_num/self.local_severity_params["population"]
        self.local_infectious *= 1 - passenger_num/self.local_severity_params["population"]
        self.local_recovered *= 1 - passenger_num/self.local_severity_params["population"]

        passenger_remote_infectious = passenger_num * prev_state["remote_infectious"] / self.remote_severity_params["population"]
        passenger_remote_exposed = passenger_remote_infectious * self.virus_params["R0"] * action[1] * self.flight_params["flight_dis"]
        if passenger_remote_infectious + passenger_remote_exposed > passenger_num:
            passenger_remote_exposed = passenger_num - passenger_remote_infectious

        self.local_infectious += passenger_remote_infectious
        self.local_exposed += passenger_remote_exposed
        self.local_susceptible += passenger_num - passenger_remote_infectious - passenger_remote_exposed

        # Outbound flight influence for remote
        self.remote_susceptible *= 1 - passenger_num/self.remote_severity_params["population"]
        self.remote_exposed *= 1 - passenger_num/self.remote_severity_params["population"]
        self.remote_infectious *= 1 - passenger_num/self.remote_severity_params["population"]
        self.remote_recovered *= 1 - passenger_num/self.remote_severity_params["population"]

        passenger_local_infectious = passenger_num * prev_state["local_infectious"] / self.local_severity_params["population"]
        passenger_local_exposed = passenger_local_infectious * self.virus_params["R0"] * action[1] * self.flight_params["flight_dis"]
        if passenger_local_infectious + passenger_local_exposed > passenger_num:
            passenger_local_exposed = passenger_num - passenger_local_infectious

        self.remote_infectious += passenger_local_infectious
        self.remote_exposed += passenger_local_exposed
        self.remote_susceptible += passenger_num - passenger_local_infectious - passenger_local_exposed

        
        # Update reward
        self.reward = 0
        self.reward -= self.flight_cost * action[0]
        self.reward += self.passenger_reward * passenger_num
        self.reward -= self.infectious_penalty * (passenger_local_infectious + passenger_remote_infectious)

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.calc_state(), self.reward, done, {}


# # Extreme test: flight = 0, occupancy = 0

# In[277]:


env = FlightEnv()

done = False
states = []
rewards = []

states.append(env.reset())

while not done:
    action = ((0, 0))
    new_state, reward, done, _ = env.step(action)
    states.append(new_state)
    rewards.append(reward)


# In[278]:


title = "Flight = 0"
plot_states(states, title)
plot_rewards(rewards, title)


# # Extreme test: flight = 1, occupancy = 1

# In[386]:


env = FlightEnv()

done = False
states = []
rewards = []

states.append(env.reset())

while not done:
    action = ((1, 1))
    new_state, reward, done, _ = env.step(action)
    states.append(new_state)
    rewards.append(reward)

env.close()


# In[281]:


title = "Flight = 1, Occupancy = 1"
plot_states(states, title)
plot_rewards(rewards, title)


# # Random action test

# In[385]:


env = FlightEnv()

done = False
states = []
actions = []
rewards = []

states.append(env.reset())

while not done:
    action = np.random.uniform(0, 1, 2)
    new_state, reward, done, _ = env.step(action)
    states.append(new_state)
    actions.append(action)
    rewards.append(reward)

env.close()


# In[293]:


title = "Random flight"
plot_states(states, title)
plot_actions(actions, title)
plot_rewards(rewards, title)


# # Q learning Algorithm

# In[579]:


# Learning paras
LEARNING_RATE = 0.1
DISCOUNT = 0.95     # gamma, how much we value future reward over current reward
EPISODES = 30000

SHOW_EVERY = 100

epsilon = 0.9
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.1


# In[583]:


# Define continuous to discrete paras
state_bins = 40     # can change to more bins
flight_unit = 2     # in action: decide on every 5 flights (can change)
seat_unit = 2       # in action: decide on every 5 seats (can change)

DISCRETE_STATE_SIZE = [state_bins] * 2
DISCRETE_ACTION_SIZE = [int(flight_assumptions["max_flight"]/flight_unit), int(flight_assumptions["full_capacity"]/seat_unit)]

print(DISCRETE_STATE_SIZE)
print(DISCRETE_ACTION_SIZE)


# In[552]:


# Convert continuous to discrete
def get_discrete_state(state):
    discrete_state = []
    discrete_state.append(state[2] * state_bins)
    discrete_state.append(state[6] * state_bins)
    return tuple(np.array(discrete_state).astype(np.int))

def get_discrete_action(action):
    discrete_action = np.array(action) * DISCRETE_ACTION_SIZE
    return tuple(discrete_action.astype(np.int))


# In[553]:


def test_qtable(q_table, title):
    env = FlightEnv()

    done = False
    states = []
    actions = []
    rewards = []

    state = env.reset()
    states.append(state)
    discrete_state = get_discrete_state(state)

    while not done:
        discrete_action = (np.argmax(np.max(q_table[discrete_state], axis = 1)), np.argmax(np.max(q_table[discrete_state], axis = 0)))
        action = np.array(discrete_action) / DISCRETE_ACTION_SIZE

        new_state, reward, done, _ = env.step(action)
        discrete_state = get_discrete_state(new_state)

        states.append(new_state)
        actions.append(action)
        rewards.append(reward)

    env.close()
    
    plot_states(states, title)
    plot_actions(actions, title)
    plot_rewards(rewards, title)


# In[584]:


# Begin learning
env = FlightEnv()
q_table = np.random.uniform(low = 0, high = 1, size = (DISCRETE_STATE_SIZE + DISCRETE_ACTION_SIZE))
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        # Use epsilon for exploration
        if np.random.random() < epsilon:
            discrete_action = (np.random.randint(0, DISCRETE_ACTION_SIZE[0]), np.random.randint(0, DISCRETE_ACTION_SIZE[1]))
        else:
            discrete_action = (np.argmax(np.max(q_table[discrete_state], axis = 1)), np.argmax(np.max(q_table[discrete_state], axis = 0)))
        action = np.array(discrete_action) / DISCRETE_ACTION_SIZE

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        # Update Q table
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + discrete_action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + discrete_action] = new_q

        discrete_state = new_discrete_state

    epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)
        
    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        avg_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")

env.close()


# In[418]:


# no epsilon exploration
title = "Q table after 2000 episodes"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)


# In[425]:


title = "Q table after 10000 episodes"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.9999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5
# reward = (1, 1, -10)


# In[563]:


title = "Q table after 30000 episodes"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.9999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5

# reward = (1, 1, -10)


# In[571]:


title = "epsilon decay = 0.999"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5

# reward = (1, 1, -10)


# In[577]:


title = "epsilon = 0.7"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5

# reward = (1, 1, -10)


# In[582]:


title = "discrete states = 40, every 2 flight, 2 passenger"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5

# reward = (1, 1, -10)


# In[585]:


# (40, 2, 2) test 2
title = "discrete states = 40, every 2 flight, 2 passenger"
test_qtable(q_table, title)
plot_ep_rewards(aggr_ep_rewards, title)

# np.save("q_table", q_table)
# np.save("ep_rewards", ep_rewards)
# np.save("aggr_ep_rewards", aggr_ep_rewards)
# q_table = np.load("q_table.npy")
# ep_rewards = np.load("ep_rewards.npy")
# aggr_ep_rewards = np.load("aggr_ep_rewards.npy",allow_pickle=True).item()

# For record: paras for this result
# epsilon = 0.9
# epsilon_decay = 0.999
# state_bins = 20
# flight_unit = 5
# seat_unit = 5

# reward = (1, 1, -10)


# In[ ]:





# # PPO: Proximal Policy Optimization Algorithms

# In[457]:


from pyforce.env import DictEnv, ActionSpaceScaler, TorchEnv
from pyforce.nn.observation import ObservationProcessor
from pyforce.nn.hidden import HiddenLayers
from pyforce.nn.action import ActionMapper
from pyforce.agents import PPOAgent
import torch


# In[464]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"

env = FlightEnv()
env = DictEnv(env)
env = ActionSpaceScaler(env)
env = TorchEnv(env).to(device)

observation_processor=ObservationProcessor(env)
hidden_layers=HiddenLayers(observation_processor.n_output)
action_mapper=ActionMapper(env,hidden_layers.n_output)

print(observation_processor)
print(hidden_layers)
print(action_mapper)


# In[466]:


agent=PPOAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path="DQN_test",
    value_lr=5e-4,
    policy_lr=5e-5
).to(device)


# In[467]:


agent.train(env,episodes=30000,train_freq=512,eval_freq=50,render=False, batch_size=64,gamma=.9,tau=.95,clip=.2,n_steps=16,entropy_coef=.001)


# In[470]:


agent


# In[486]:


def test_agent(agent, env, **kwargs):
    # this env is the env in torch

    done = False
    states = []
    actions = []
    rewards = []

    state = env.reset()
    states.append(state)

    while not done:
        action, action_info = agent.get_action(state, True, kwargs)

        new_state, reward, done, _ = env.step(action)

        states.append(new_state)
        actions.append(action)
        rewards.append(reward)
        
        state = new_state
    
    return states, actions, rewards


# In[513]:


# Visualize "agent" result
states_ppo, actions_ppo, rewards_ppo = test_agent(agent, env)

title = "PPOAgent"

states_new = []
for each in states_ppo:
    states_new.append(list(np.array(each['state'])[0]))
plot_states(states_new, title)

actions_new = []
for each in actions_ppo:
    actions_new.append(list(np.array(each['action'])[0]))
plot_actions(actions_new, title)

rewards_new = []
for each in rewards_ppo:
    rewards_new.append(np.array(each)[0][0])
plot_rewards(rewards_new, title)


# In[514]:


tf_env = copy.deepcopy(env)


# # Evaluation

# In[ ]:




