from dataclasses import dataclass, field
from typing import Callable
import time

import numpy as np
from tqdm import tqdm
from .trees import run


MIN_BOUND = 1
MAX_BOUND = 10
FEATURES = 'max_depth', 'min_samples_split', 'max_len'
NUM_FEATURES = len(FEATURES)
NUM_GENERATIONS = 100
NUM_POPULATION = 20


def initialize(population_size: int, n_features: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    shape = (population_size, n_features)
    result = np.random.randint(MIN_BOUND, MAX_BOUND, shape)
    np.random.shuffle(result)
    return result


def score(chromosome: np.ndarray) -> float:
    max_depth, min_samples_split, max_len = chromosome
    mse = run(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_len=max_len
    )
    return mse

def fitness_score(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    population: after initialization
    """

    population_list = list(map(lambda arr: list(map(int, arr)), population))
    scores = list(tqdm(map(score, population_list), total=len(population_list)))
    scores = np.array(scores)

    # obtendo indices que ordenam pelo maior score
    inds = np.argsort(scores) # crescente
    # inds = np.argsort(scores)[::-1] # decresente

    ordered_scores = scores[inds]
    ordered_population = population[inds, :]

    return ordered_scores, ordered_population


def selection(population: np.ndarray, n_parents: int | None = None) -> np.ndarray:
    """
    population: after score ordering
    """
    # por padrão, metade da população passará a próxima etapa
    if n_parents is None:
        n_chromo = population.shape[0]
        n_parents = n_chromo// 2
    return population[:n_parents]

CrossoverStrategy = Callable[[np.ndarray, np.ndarray], 'tuple[np.ndarray,np.ndarray]']

def uniform_strategy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """get genes from both parents in a uniform way
    ex: from parents AAAAA and BBBBB
        ABAAB
        BABBA
    """
    cond = np.random.randint(0, 2, a.shape[-1], dtype=bool)
    return np.where(cond, a, b), np.where(~cond, a, b)


def singlepoint_strategy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """split each chromosome in 2 parts and combine to make 2 children
    ex: from parents AAAAA and BBBBB
        AABBB
        BBAAA
    """
    cond = np.zeros(a.shape, bool)
    cond[:len(a)//2] = True
    return np.where(cond, a, b), np.where(~cond, a, b)


def multipoint_strategy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """split each chromosome in 3 parts and combine to make 2 children
    ex: from parents AAAAA and BBBBB
        AABAA
        BBABB
    """
    cond = np.zeros(a.shape, bool)
    third = len(a)//3
    cond[third:2*third] = True
    return np.where(cond, a, b), np.where(~cond, a, b)


def random_singlepoint_strategy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cond = np.zeros(a.shape, bool)
    cond[:np.random.randint(2, len(a))] = True
    return np.where(cond, a, b), np.where(~cond, a, b)


def crossover(
    population: np.ndarray,
    strategy: CrossoverStrategy = uniform_strategy
) -> np.ndarray:
    """Genetic Algorithm Crossover
    
    population: população ordenada pelo melhor fitness
    """
    next_population = list(population)

    # itera de 2 em 2, se tiver população impar irá pular o último
    for i in range(0, len(population) -1, 2):
        # selecionando indices que serão pertencentes a cada parente
        children = strategy(population[i], population[i + 1])

        # adiciona a crianças à população mantendo também os pais
        next_population.extend(children)

    # se for ímpar, o último cromosomo não teve reprodução
    if len(population) % 2 != 0:
        # misturando genes do melhor e pior cromosomo
        children = strategy(population[0], population[-1])
        next_population.extend(children)

    return np.array(next_population)

def mutation(population: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
    """
    population: after crossover
    """
    num_features = population.shape[1]
    num_mutations = int(mutation_rate * num_features)

    next_population = []
    for chromo in population:

        # Calculando quais genes do cromosomo serão mutados
        indexes = np.random.randint(0, num_features -1, size=num_mutations)
        
        # Realizando Mutação no cromosomo
        chromo[indexes] = np.clip(
            chromo[indexes] + np.random.randint(-2, 3, len(indexes)),
            MIN_BOUND,
            MAX_BOUND
        )
        next_population.append(chromo)

    return np.array(next_population)

@dataclass
class LifeBook:
    generation: list[int] = field(default_factory=list)
    best_score: list[float] = field(default_factory=list)
    best_chromo: list[np.ndarray] = field(default_factory=list)
    populations: list[np.ndarray] = field(default_factory=list)
    time_passed: list[float] = field(default_factory=list)
    total_time_passed: float = field(default=0.0)

    def write(self,
        generation: int,
        best_score: float,
        best_chromo: np.ndarray,
        population: np.ndarray,
        time_passed: float
    ) -> None:
        self.generation.append(generation)
        self.best_score.append(best_score)
        self.best_chromo.append(best_chromo)
        self.populations.append(population)
        self.time_passed.append(time_passed)


def execute(
    n_generations: int = 10,
    population_size: int = 20,
    n_features: int = NUM_FEATURES # -> epochs, batch_size, neurons, activation, optimization
):
    # no inicio nada existia, então criei o tempo...
    begin = time.time()

    # então eu criei o mundo...
    population = initialize(population_size, n_features)

    # iniciei a escritura do livro da vida...
    lifebook = LifeBook()

    # julguei os primeiros da existencia para escolher um rei...
    score, population = fitness_score(population)
    best_score = score[0]
    best_chromo = population[0]

    # planejei o fim do mundo e quantas gerações devem existir...
    for i in range(n_generations):
        begin_generation = time.time()

        # matei os que não mereciam viver e preservei os mais fieis...
        population = selection(population)

        # arrangei os casamentos e deixei que tivessem filhos...
        population = crossover(population)

        # mudei a aparencia dos mais feios, segundo a minha vontade...
        population = mutation(population)

        # então eu julguei a todos...
        score, population = fitness_score(population)
        if score[0] < best_score:
            best_score = score[0]
            best_chromo = population[0]
        
        # ao fim de cada geração anotei tudo no livro da vida...
        lifebook.write(
            generation=i,
            best_score=best_score,
            best_chromo=best_chromo,
            population=population,
            time_passed=time.time() - begin_generation
        )

        # elevei meu fiel preferido diante dos outros...
        print('Geração', i,': Melhor Pontuação:', best_score, 'em:', lifebook.time_passed[-1], 'feito por:', best_chromo)

    # assim o mundo acabou...
    end = time.time()
    lifebook.total_time_passed = end - begin
    return lifebook



if __name__ == "__main__":
    import pickle
    NUM_GENERATIONS = 50

    # exp5 = main(NUM_GENERATIONS, 5)
    # pickle.dump(exp5, open('artifacts/exp5.pkl', 'wb'))
    
    # exp10 = main(NUM_GENERATIONS, 10)
    # pickle.dump(exp10, open('artifacts/exp10.pkl', 'wb'))
    
    # exp15 = main(NUM_GENERATIONS, 15)
    # pickle.dump(exp15, open('artifacts/exp15.pkl', 'wb'))

    # exp20 = main(NUM_GENERATIONS, 20)
    # pickle.dump(exp20, open('artifacts/exp20.pkl', 'wb'))
    obj: LifeBook = pickle.load(open('artifacts/exp20.pkl', 'rb'))

    # exp25 = main(NUM_GENERATIONS, 25)
    # pickle.dump(exp25, open('artifacts/exp25.pkl', 'wb'))

    # exp30 = main(NUM_GENERATIONS, 30)
    # pickle.dump(exp30, open('artifacts/exp30.pkl', 'wb'))

    print('wait')

