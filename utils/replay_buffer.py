import numpy as np

'''
naive experience memory
'''
class ExperienceMemory():
    '''
    #if self.memory._len() < self.batch_size:
    #   return

    #states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

    #self.memory.add(state, next_state, reward, action, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.next_idx = 0

    def _len(self):
        return len(self.storage)

    def add(self, data):
        if self.max_size >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, reward, action, done = [], [], [], [], []

        for i in idx:
            s, next_s, r, a, d = self.storage[i]

            state.append(np.array(s, copy=False))
            next_state.append(np.array(next_s, copy=False))
            reward.append(np.array(r, copy=False))
            action.append(np.array(a, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(next_state), np.array(reward), np.array(action), np.array(done)

        
        # %%
if __name__ == "__main__":
    memory = ExperienceMemory(30000)
    print(memory._len())
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))
    memory.add((1, 2, 3, [4, 5, 6, 7], 8))

    print(memory.sample(1))