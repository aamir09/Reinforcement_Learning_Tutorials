import numpy as np
import tqdm

np.random.seed(101)
class Bandit:
    def __init__(self, k:int, steps:int, trials:int):
        self.k = k
        self.steps = steps
        self.n_trials = trials
        self.q_a_list = []
        self.rewards_list = []
        
    def calcaulate_value(self, sum, n):
        # handling the case where actions haven't been taken before; n=0
        if n==0:
            return 0
        return sum/n
        
    def trial(self, policy):
        ## Initiate dictionaries to calculate sum of rewards for each action/bandit
        values = {k:v for k,v in zip(range(self.k),[0]*self.k)}
        
        ## Initiate dictionary for keeping record of number of times an action is taken 
        action_taken = {k:v for k,v in zip(range(self.k),[0]*self.k)}
        
        ## Initialize the actual values for actions 
        q_a = np.array([np.random.standard_normal(size=self.k)]).flatten()
        
        optimal_action = np.argmax(q_a)
        
        ## Initialize rewards based on the actual values of action
        rewards = np.random.normal
        
        ## For record keeping
        self.q_a_list.append(q_a)
        self.rewards_list.append(rewards)
        
        q_t_list = []
        
        is_optimal = []
        
        for _ in range(self.steps):
            
            ## We will use the sample-average method to calculate Q(a)_t
            #Calculate the q-value for all actions prior step t
            Q_a__t = [self.calcaulate_value(sum=item[0],n=item[1]) for item in 
                      zip(values.values(),action_taken.values())]
            
            #select an action according to the policy 
            action, q_t = policy(q_vals=Q_a__t)
            q_t_list.append(q_t)
            
            if action == optimal_action:
                is_optimal.append(1)
            else:
                is_optimal.append(0)
            
            #Add to the sum of the rewards for that action
            #and the number of times the action is taken
            action_taken[action]+=1
            values[action] = values[action] + (1/action_taken[action])*(rewards(scale=0.01)-values[action])
            
            
        return is_optimal, action_taken, np.array(q_t_list)
    
    def simulate(self, policy):
        
        print("""
              Initial conditions:
              Number of Trials = {trials}
              Number of Steps per Trial = {steps}
              NUmber of Bandit Arms = {k}
              """.format(trials=self.n_trials, steps=self.steps, k=self.k))
        
        is_optimals = np.zeros(shape=(self.n_trials,self.steps))
        actions = {}
        q_matrix = np.zeros(shape=(self.n_trials,self.steps))
        for t in tqdm.tqdm(range(self.n_trials),colour="blue"):
            s, a, q_t = self.trial(policy=policy)
            is_optimals[t, :] = s
            actions[t] = a
            q_matrix[t, :] = q_t
        
        return is_optimals, actions, q_matrix
    
    
    

        
        
        
            
            
        
        