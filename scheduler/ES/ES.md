
# reference
- ES: Evolutionary Search
- https://github.com/MorvanZhou/Evolutionary-Algorithm
- https://mofanpy.com/tutorials/machine-learning/evolutionary-algorithm/

# outline
- genetic algorithm
  - basic GA
  - match phrase
  - travel sales
  - find path
  - microbial GA
- evolution strategy
  - basic ES
  - (1+1) - ES
  - NES, natural evolution strategy
- neural nets
  - NEAT for supervised-learning
  - NEAT for reinforcement-learning
  - Distributed ES with Neural nets

# Evolutionary Algorithm
## init DNA
## encode DNA
## Generation
- fitness
- best
- crossover
- mutate
## decode DNA

# Evolutionary Strategy
## diff
- No more need
  - encode DNA
  - decode DNA
- DNA
  - real number; binary
  - DNA[value, mutate_strength]; DNA[value]
    - value
      - satisfy normal distribution
      - clip DNA_BOUND
    - mutate_strength
      - decrease gradually
- fitness
  - MaxTopN(pop + kids)
- crossover
  - parent: random; best
- mutate
  - gassian distribution; random change

## (1+1)-ES
- Tutorial CMA-ES: evolution strategies and covariance matrix adaptation
  - https://www.researchgate.net/publication/220740663_CMA-ES_evolution_strategies_and_covariance_matrix_adaptation
- definition: (p/n +, c)
  - n: topN in p
  - p: the number of population
  - c: the number of child
  - +: concate + fitness(topN)
  - ,: fitness(p, c)
- mutate_strength
  - 1/5
  - god child: ms *= 2.028
  - bad child: ms *= -0.507

## NES
