import typing
from typing import Tuple
import random
import numpy as np
from utils.sum_tree import SumTree

class PrioritizedMemory:
    def __init__(self, capacity:int):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.alpha = 0.6
        self.alpha_decretement_per_sampling = 0.0001
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    # get priority from TD error in memory
    def _get_priority(self, error:float) -> float:
        return (np.abs(error) + self.e) ** self.alpha

    def _len(self) -> int:
        return self.tree.n_entries

    # add trajectory in prioritiy memory
    def add(self, error:float, sample:list) -> None:
        p = self._get_priority(error)
        self.tree.add(p, sample)

    # get samples from priority memory according mini batch size n
    def sample(self, n:int) -> list:
        states = np.array([], dtype = np.float32)
        next_states = np.array([], dtype = np.float32)
        rewards = np.array([], dtype = np.float32)
        actions = np.array([], dtype = np.float32)

        dones = np.array([], dtype = np.int16)
        idxs = np.array([], dtype = np.int16)
        priorities = np.array([], dtype = np.float32)

        segment = self.tree.total() / n

        self.alpha = np.max([0., self.alpha - self.alpha_decretement_per_sampling])
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        data = []
        idx = 0
        p = 0.0
        for iteration in range(n):
            a = segment * iteration
            b = segment * (iteration + 1)

            while True:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if not isinstance(data, int):
                    break
            
            if(iteration == 0):
                states = np.hstack((states, data[0]))
                next_states = np.hstack((next_states, data[1]))
                rewards = np.hstack((rewards, data[2]))
                actions = np.hstack((actions, data[3]))

                dones = np.hstack((dones, data[4]))
                idxs = np.hstack((idxs, idx))
                priorities = np.hstack((priorities, p))
            
            else:
                states = np.vstack((states, data[0]))
                next_states = np.vstack((next_states, data[1]))
                rewards = np.hstack((rewards, data[2]))
                actions = np.vstack((actions, data[3]))

                dones = np.hstack((dones, data[4]))
                idxs = np.hstack((idxs, idx))
                priorities = np.hstack((priorities, p))

        sampling_probabilities = priorities / self.tree.total()
        
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        
        min_probability = self._get_priority(self.e) / self.tree.total()
        max_weights = np.power(self.tree.n_entries * min_probability, -self.beta)

        is_weight /= max_weights

        return states, next_states, rewards, actions, dones, idxs, is_weight

    # update priority of prioritized memory
    def update(self, idx:int, error:float) -> None:
        p = self._get_priority(error)
        self.tree.update(idx, p)

