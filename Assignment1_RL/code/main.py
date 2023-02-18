


from Taxi import Taxi
import matplotlib.pyplot as plt
import gym
import numpy as np
import random


def main():
    gama = 0.95
    # number of environments to explore prior to applying policy iteration
    n_env = 50

    # call taxi class with gama  and n_env for the number of environments 
    taxi = Taxi(gama,n_env)
    taxi.env_learn()

    # check_states array where the index represents the state and the value represents the states value  
    check_states = taxi.value_function

    # plot the mean expected return of all n_env given environments
    fig, ax = plt.subplots()
    ax.plot(list(taxi.policy_exp.keys()),list(taxi.policy_exp.values()))
    ax.set(xlabel='Iteration', ylabel='Mean Expected Reward',title='Policy Evaluation')
    ax.locator_params(integer=True)
    ax.grid()
    plt.show()
    print()
    print("########### Simulation  ############")
    print()
    
    # run  simulation
    simulation(taxi)
    
    print()
    print("########### State Values  ############")
    print()
    
    for i in range(len(check_states)):
        print(" State: "+ str(i) + ",Generated Value: "+str(check_states[i]))
        

# decode the given state to a an array 
def decode( i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return list(reversed(out))

# runs a simulation using the optimal policy found from the policy iteration algorithm
def simulation(taxi, render=True):
    # reset environment
    env = gym.make('Taxi-v3')
    env.reset()
    
    # maintain 3 lists for printing info
    policies = []
    states = []
    rewards = []

    total_reward = 0
    reward = 0
    
    i = 0
    
    # start state
    current_state = env.s
    states.append(current_state)
    
    # limit the simulation to 200 iterations (steps)
    while i<200 and reward!=20:
        
        # plot the environment
        if render:
            env.render()
        
        # if the state is known, get the optimnal policy else find a random policy
        if current_state in taxi.known_states:                
            policy =taxi.policy_function[current_state]
        else:
            policy = random.choice([0,1,2,3,4,5])
                                
        # take the optimal step, and get the  next state, the reward
        next_state, reward, done, info = env.step(policy)
        
        # update the lists
        total_reward += round(reward*pow(taxi.gama, i),2)        
        rewards.append(reward)
        policies.append(policy)
        states.append(next_state)

        #next_state = taxi.transition_function[current_state,policy].copy()
        current_state = next_state
        i += 1
        
    print()        
    print("Total steps: "+str(i))
    print("Total rewards: "+str(sum(rewards)))
    print("Total discounted rewards: "+str(round(total_reward,2)))
    print()
    
    locations = {0: "0,0",1: "0,4", 2: "4,0", 3: "4,3"}
    step_direction = {0: "move south",
                     1: "move north",
                     2: "move east",
                     3: "move west",
                     4: "pickup passenger",
                     5: 'drop off passenger'}
    
    for i in range(len(states)-1):
                
        decoded_state = decode(states[i])
        reward = rewards[i]
        policy = step_direction[policies[i]]
        
        # get the taxis and the destinations locations
        taxi_loc = str(decoded_state[0])+","+str(decoded_state[1])
        destination_loc = locations[decoded_state[3]]

        # if the passenger is not the in taxi then get the initial location else get the taxis location
        if decoded_state[2]!=4:           
            passenger_loc = locations[decoded_state[2]]
        else:
            passenger_loc = taxi_loc

        print(str(i+1) + ". " + taxi_loc + " , " + passenger_loc + " , "
              + destination_loc + " , " + policy + " , " + str(reward))
        


if __name__ == "__main__":
    main()





