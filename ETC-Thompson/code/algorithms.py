from env_MAB import *
from functools import lru_cache

def random_argmax(a):
    '''
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    '''
    return np.random.choice(np.where(a == a.max())[0])


class Explore():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        current_record = self.MAB.get_record()
        num_pulls = current_record[:,0] + current_record[:,1] # get the number of times each arm has been pulled
        arm = random_argmax(-num_pulls) # select the arm with the least number of pulls
        self.MAB.pull(arm)

class Greedy():
    def __init__(self, MAB):
        self.MAB = MAB
        self.t = 0

    def reset(self):
        self.MAB.reset()
        self.t = 0

    def play_one_step(self):
        current_record = self.MAB.get_record()
        num_pulls = current_record[:,0] + current_record[:,1] # get the number of times each arm has been pulled
        if self.t >= self.MAB.get_K(): # if all arms have been pulled at least once
            current_rewards = current_record[:,1] / num_pulls # get the reward for each arm
            arm_to_pull = random_argmax(current_rewards) # get arm with highest reward
        else: # otherwise, pull an arm that has not been pulled yet
            arm_to_pull = self.t
        self.MAB.pull(arm_to_pull)
        self.t += 1


class ETC():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.N_e = np.floor((self.MAB.get_T() * np.sqrt(np.log(2*self.MAB.get_K()/self.delta) / 2) / self.MAB.get_K()) ** (2/3))
        self.empirical_record = np.zeros((self.MAB.get_K(),2))

    def reset(self):
        self.MAB.reset()
        self.empirical_record = np.zeros((self.MAB.get_K(),2))

    def play_one_step(self):
        current_record = self.MAB.get_record()
        num_pulls = current_record[:,0] + current_record[:,1] # get the number of times each arm has been pulled
        if np.all(num_pulls >= self.N_e): # if all arms have been pulled at least N_e times
            empirical_means = self.empirical_record[:,1] / self.N_e
            arm_to_pull = random_argmax(empirical_means) # get arm with highest empirical mean
            self.MAB.pull(arm_to_pull)
        else:
            # choose randomly among the arms that have not been pulled N_e times yet
            arm_to_pull = np.random.choice(np.where(num_pulls < self.N_e)[0])
            reward = self.MAB.pull(arm_to_pull)
            self.empirical_record[arm_to_pull, reward] += 1 # update the empirical record

            
class Epgreedy():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.t = 0

    def reset(self):
        self.MAB.reset()
        self.t = 0
    
    def play_one_step(self):
        """
        First pull each arm once. Then for each later t >= K:
            With probability ep_t, randomly choose an arm to pull (explore).
            With probability 1 - ep_t, pull the greedy arm (exploit),
        where ep_t = (K ln(t) / t) ^ (1/3) and ties are broken at random.
        """
        if self.t < self.MAB.get_K():
            self.MAB.pull(self.t) # pull each arm once
        else: # for each later t >= K
            current_record = self.MAB.get_record()
            num_pulls = current_record[:,0] + current_record[:,1] # get the number of times each arm has been pulled
            current_rewards = current_record[:,1] / num_pulls # get the reward for each arm
            greedy_arm = random_argmax(current_rewards) # get the greedy arm, breaking ties at random
            ep_t = (self.MAB.get_K() * np.log(self.t) / self.t) ** (1/3)
            
            if np.random.rand() < ep_t: # explore with probability ep_t
                arm_to_pull = np.random.choice(self.MAB.get_K()) # choose randomly among all arms
            else: # exploit with probability 1 - ep_t
                arm_to_pull = greedy_arm
            
            self.MAB.pull(arm_to_pull) # pull the arm

        self.t += 1


class UCB():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()

    def play_one_step(self): 
        current_record = self.MAB.get_record()
        num_pulls = current_record[:,0] + current_record[:,1]
        # get the reward for each arm, and set the reward to infinity for arms that have not been pulled yet
        current_rewards = np.where(num_pulls > 0, current_record[:,1] / num_pulls, np.inf)
        # get the upper confidence bound for each arm
        ucb = current_rewards + np.sqrt(np.log(self.MAB.get_T() * self.MAB.get_T() / self.delta) / (num_pulls))
        arm_to_pull = random_argmax(ucb)
        self.MAB.pull(arm_to_pull)


class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB
        self.alpha = np.ones((self.MAB.get_K(),))
        self.beta = np.ones((self.MAB.get_K(),))


    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.alpha = np.ones((self.MAB.get_K(),))
        self.beta = np.ones((self.MAB.get_K(),))
 

    def play_one_step(self):
        '''
        Implement one step of the Thompson sampling algorithm. 
        '''
        thetas = np.random.beta(self.alpha, self.beta) # sample a reward for each arm
        arm_to_pull = random_argmax(thetas) # get the arm with the highest reward, breaking ties at random 
        reward = self.MAB.pull(arm_to_pull) # pull the arm
        # update the parameters
        self.alpha[arm_to_pull] += reward
        self.beta[arm_to_pull] += 1 - reward
        

class Gittins_index():
    def __init__(self, MAB, gamma=0.90, epsilon=1e-4, N=100):
        self.MAB = MAB
        self.gamma = gamma
        self.epsilon=epsilon
        self.N = N
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())
        gi = self.compute_gittins_index(0)
        self.gittins_indices = np.ones(self.MAB.get_K())*gi


    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.lower_bound = 0
        self.upper_bound = 1/(1-self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())
        gi = self.compute_gittins_index(0)
        self.gittins_indices = np.ones(self.MAB.get_K())*gi
    
    @lru_cache(maxsize=None)
    def calculate_value_oab(self, successes, total_num_samples, lambda_hat, stage_num=0):
        '''
        Helper function for calculating the OAB value. Recursive function
        '''
        p = successes / total_num_samples # calculate the probability of success
        q = 1 - p # calculate the probability of failure

        if stage_num == self.N:
            return ((self.gamma ** self.N) / (1 - self.gamma)) * max(0, p - lambda_hat/(1 - self.gamma))
        else:
            value_oab = max(
                p - lambda_hat / (1 - self.gamma) \
                  + self.gamma * (p * self.calculate_value_oab(successes + 1, total_num_samples + 1, lambda_hat, stage_num + 1) \
                                + q * self.calculate_value_oab(successes, total_num_samples + 1, lambda_hat, stage_num + 1)),
                0
            )
            return value_oab
    
    def compute_gittins_index(self, arm_index):
        '''
        Calibration for Gittins Index (Algorithm 1)
        '''
        ub = self.upper_bound
        lb = self.lower_bound
        current_successes = 1 + self.MAB.get_record()[arm_index, 1]
        current_failures = 1 + self.MAB.get_record()[arm_index, 0]
        current_total_num_samples = current_successes + current_failures
        while ub - lb > self.epsilon:
            lambda_hat = (ub + lb) / 2
            if self.calculate_value_oab(current_successes, current_total_num_samples, lambda_hat) > 0:
                lb = lambda_hat
            else:
                ub = lambda_hat
        self.gittins_indices[arm_index] = ub  
        return ub      

    def play_one_step(self):
        '''
        Select the arm with the highest Gittins Index and about its Gittins Index based on the value return by pull 
        '''
        # calculate the Gittins Index for each arm
        arm_to_pull = random_argmax(self.gittins_indices)
        self.compute_gittins_index(arm_to_pull)
        print(arm_to_pull)
        self.MAB.pull(arm_to_pull)
