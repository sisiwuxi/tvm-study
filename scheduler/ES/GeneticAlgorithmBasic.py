"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://mofanpy.com/tutorials/

DNA: binary
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 20 # 200
X_BOUND = [0, 5]         # x upper and lower bounds


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    # mother: parent
    # father: pop[i_]
    # which points in father's DNA
    if np.random.rand() < CROSS_RATE:
        # import pdb;pdb.set_trace()
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        mask = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)   # choose crossover points
        parent[mask] = pop[i_, mask]                            # mating and produce one child
    return parent


def mutate(child):
    # mutate per point in DNA
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            # import pdb;pdb.set_trace()
            child[point] = 1 if child[point] == 0 else 0
    return child

if __name__ == "__main__":
    # 0,1 -> float
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA
    
    plt.ion()       # something about plotting
    x = np.linspace(*X_BOUND, N_GENERATIONS)
    plt.plot(x, F(x))
    # import pdb;pdb.set_trace()

    for _ in range(N_GENERATIONS):
        F_values = F(translateDNA(pop))    # compute function value by extracting DNA

        # something about plotting
        if 'sca' in globals(): sca.remove()
        sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

        # GA part (evolution)
        # higher --> better
        fitness = get_fitness(F_values)
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        # higher fitness -> more opportunity
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child       # parent is replaced by its child

    plt.ioff(); plt.show()
