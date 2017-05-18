"""
Implements genetic algorithm. Exercise 4 Inteligent Data Processing.
"""
# http://pythonhosted.org/bitstring/creation.html
import bitstring
import logging
import random

logger = logging.getLogger(__name__)

class Population(object):
    """
    Represents population
    """
    population = []
    population_count = None

    def __init__(self, start, stop, population_count):
        """
        Return population(list) of random numbers in range start, stop
        Each value is an special object which handle bin, float and int repr.
        :param population_count:
        :return:
        """
        self.population_count = population_count

        for i in xrange(population_count):
            self.population.append(
                bitstring.BitArray(
                    float=random.uniform(start, stop), length=64, ),
            )
        logger.debug(
            'First generation of population: {}'.format(self.population)
        )

    def mutate(self, mutate_prob=0.01):
        """
        Mutate individuals in population
        mutate_prob of population mutate in single generation
        :return:
        """
        mutations_schema = {
            '0': '1',
            '1': '0',
        }
        for i, ind in enumerate(self.population):
            mutate_random = random.uniform(0, 1)
            if mutate_prob >= mutate_random:
                random_pos = random.randint(0, 63)
                ind_binary = list(ind.bin)
                ind_binary[random_pos] = mutations_schema[
                    ind_binary[random_pos]
                ]
                self.population[i] = bitstring.BitArray(
                    bin="".join(ind_binary), )

    def crossover(self, cross_propability=0.2):
        """
        Crossover individuals in population
        :return:
        """
        def find_parents(parents_a, parents_b, ):
            random_parent = random.randint(0, self.population_count-1)
            if (
                random_parent not in parents_a and
                random_parent not in parents_b
            ):
                return random_parent
            else:
                return find_parents(parents_a, parents_b)

        # store indexes of parents in self.population
        parents_a = []
        parents_b = []

        for i, ind in enumerate(self.population):
            crossover_random = random.uniform(0, 1)
            if cross_propability >= crossover_random:
                parent_a = find_parents(parents_a, parents_b)
                parent_b = find_parents(parents_a, parents_b)
                parents_a.append(parent_a)
                parents_b.append(parent_b)

        for j, parent in enumerate(parents_a):
            parent_b_index = parents_b[j]
            random_pos = random.randint(1, 63)
            bin_a = self.population[parent].bin
            bin_b = self.population[parent_b_index].bin
            new_bin_a = bin_a[0: random_pos] + bin_b[random_pos:]
            new_bin_b = bin_b[0: random_pos] + bin_a[random_pos:]
            self.population[parent] = bitstring.BitArray(bin="".join(new_bin_a), )
            self.population[parent_b_index] = bitstring.BitArray(bin="".join(new_bin_b), )

    def selection(self, func,):
        """
        Select only the best individuals
        Get sum of func(individual) for all individuals
        Calculate weight for each individual where p(x) = func(x) / sum of func
        Rulette selection for calculated weights
        :param func: test function
        :return:
        """
        def weighted_random_choice(choices):
            """
            Rulette selection
            :param choices:
            :return:
            """
            max = sum(choices.values())
            pick = random.uniform(0, max)
            current = 0
            for key, value in choices.items():
                current += value
                if current > pick:
                    return bitstring.BitArray(
                        float=key, length=64,
                    )

        choices = {}
        for individual in self.population:
            # assign value of test function to individual value
            choices[individual.float] = func(individual.float)

        new_population = []
        while len(new_population) != len(self.population):
            selected_ind = weighted_random_choice(choices)
            if selected_ind:
                new_population.append(
                    selected_ind,
                )

        self.population = new_population

DOMAIN = (0.5, 2.5)
ITERATION_NO = 100
def test_function(x):
    return x + 0.5

population = Population(0.5, 2.5, 20)
for i in xrange(0, ITERATION_NO):
    population.mutate()
    population.crossover()
    population.selection(test_function)

print population.population[0].float
print population.population[50].float
print population.population[99].float
