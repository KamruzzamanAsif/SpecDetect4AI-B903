import numpy as np

#make dummy data
data = [i for i in range(1000)]
# np.random.seed(42)
def set_seed(seed):
    np.random.seed(seed)

def train():
    # set_seed(42)
    a = np.random.shuffle(data)



if __name__ == "__main__":
    train()
    print(data[:10])  # Print first 10 elements to verify shuffling

