import numpy as np
import typing

class SumTree:
    write = 0

    def __init__(self, capacity:int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.datas = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    
    # update to the root note
    def _propagate(self, idx:int, change:float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx:int, s:float) -> int :
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # return sum of tree
    def total(self) -> float:
        return self.tree[0]

    # store priority and sample
    def add(self, p:float, data:list) -> None:
        idx = self.write + self.capacity -1
        
        self.datas[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx:int, p:float) -> None:
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s:float) -> tuple :
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.datas[dataIdx])