import typing
from typing import Tuple
from nptyping import NDArray
import random
import numpy as np
from .sum_tree import SumTree

class PrioritizedMemory:
    def __init__(self, capacity:int, alpha, beta):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.alpha = alpha
        self.alpha_decretement_per_sampling = 0.0001
        self.beta = beta
        self.beta_increment_per_sampling = 0.0001

    # get priority from TD error in memory
    def _get_priority(self, error:float) -> float:
        return np.abs(error) ** self.alpha

    # add trajectory in prioritiy memory
    def add(self, experience:tuple) -> None:
        state, action, return_g = experience

        if return_g > 0:
            p = self._get_priority(return_g + self.e)
        else:
            p = self._get_priority(self.e)

        self.tree.add(p, list([state, action, return_g])) # 확인 필요

    # get samples from priority memory according mini batch size n
    def sample(self, batch_size:int):
        states = np.array([], dtype=np.float64)
        actions = np.array([], dtype=np.int32)
        returns_g = np.array([], dtype=np.float64)

        priorities = np.array([], dtype=np.float64)
        indices = np.array([], dtype=np.int)

        segment = self.tree.total() / batch_size

        self.alpha = np.max([0., self.alpha - self.alpha_decretement_per_sampling])
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        data = []
        idx = 0
        p = 0.0
        for iteration in range(batch_size):
            a = segment * iteration
            b = segment * (iteration + 1)

            while True:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if not isinstance(data, int):
                    break

            if(iteration == 0):
                states = np.hstack((states, data[0]))
                actions = np.hstack((actions, data[1]))
                returns_g = np.hstack((returns_g, data[2]))
                
                priorities = np.hstack((priorities, p))
                indices = np.hstack((indices, idx))

            else:

                states = np.vstack((states, data[0]))
                actions = np.vstack((actions, data[1]))
                returns_g = np.hstack((returns_g, data[2]))

                priorities = np.hstack((priorities, p))
                indices = np.hstack((indices, idx))

        sampling_probabilities = priorities / self.tree.total()
        # print('sampling_probabilities : ', sampling_probabilities)
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # print('wegiths : ', weights)


        min_probability = self._get_priority(self.e) / self.tree.total()
        max_weights = np.power(self.tree.n_entries * min_probability, -self.beta)

        weights /= max_weights

        return states, actions, returns_g, indices, weights

    # update priorities of prioritized memory
    def update(self, indices, returns_g) -> None:
        for return_g, idx in zip(returns_g, indices):
            # print('in update : ', idx)
            if return_g > 0:
                updated_p = self._get_priority(return_g + self.e)
            else:
                updated_p = self._get_priority(self.e)
            
            self.tree.update(idx, updated_p)