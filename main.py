"""
Implements genetic algorithm. Exercise 4 Inteligent Data Processing.
"""
# http://pythonhosted.org/bitstring/creation.html
import bitstring
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import random


logger = logging.getLogger(__name__)

class Population(object):
    """
    Represents population
    """
    avg = None
    population = []
    population_count = None
    genes_length = 64
    start = None
    stop = None

    def __init__(self, start, stop, population_count):
        """
        Return population(list) of random numbers in range start, stop
        Each value is an special object which handle bin, float and int repr.
        :param population_count:
        :return:
        """
        self.population_count = population_count
        self.start = start
        self.stop = stop
        unique_inds = []
        while len(self.population) <= population_count:
            random_float = random.uniform(start, stop)
            if not random_float in unique_inds:
                unique_inds.append(random_float)
                self.population.append(
                    bitstring.BitArray(
                        float=random_float,
                        length=self.genes_length, ),
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
                random_pos = random.randint(0, self.genes_length - 1 )
                ind_binary = list(ind.bin)
                ind_binary[random_pos] = mutations_schema[
                    ind_binary[random_pos]
                ]
                mutated = bitstring.BitArray(
                    bin="".join(ind_binary), )
                # test if it's number
                if mutated.float == mutated.float:
                    self.population[i] = mutated

    def crossover(self, cross_propability=0.3):
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

        while len(parents_a) < len(self.population) * cross_propability:
            crossover_random = random.uniform(0, 1)
            if cross_propability >= crossover_random:
                parent_a = find_parents(parents_a, parents_b)
                parent_b = find_parents(parents_a, parents_b)
                parents_a.append(parent_a)
                parents_b.append(parent_b)

        for j, parent in enumerate(parents_a):
            parent_b_index = parents_b[j]
            random_pos = random.randint(1, self.genes_length)
            bin_a = self.population[parent].bin
            bin_b = self.population[parent_b_index].bin
            new_bin_a = bin_a[0: random_pos] + bin_b[random_pos:]
            new_bin_b = bin_b[0: random_pos] + bin_a[random_pos:]
            a_children = bitstring.BitArray(bin="".join(new_bin_a), )
            b_children = bitstring.BitArray(bin="".join(new_bin_b), )

            # nan test, if equal it means it's number
            if a_children.float == a_children.float:
                self.population[parent] = a_children
            if b_children.float == b_children.float:
                self.population[parent_b_index] = b_children

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
                        float=key, length=self.genes_length,
                    )

        choices = {}
        for individual in self.population:
            # assign value of test function to individual value
            if self.start <= individual.float <= self.stop:
                choices[individual.float] = func(individual.float)

        new_population = []
        loop_counter = 0
        while len(new_population) != len(self.population):
            # for debugging
            loop_counter += 1
            if loop_counter > 100000:
                break

            selected_ind = weighted_random_choice(choices)
            selected_float = selected_ind.float
            new_population.append(
                selected_ind,
            )

        self.avg = sum(choices.values()) / len(choices)
        self.population = new_population


def test_function(x):
    if x > 2.5:
        pass
    return ((math.e ** x) * np.sin(10 * np.pi * x) + 1) / x

DOMAIN = (0.5, 2.5)
ITERATION_NO = 50
POPULATION_COUNT = 20
avg_x = []
avg_y = []

def main(domain, iterations, test_function, inds_no):
    population = Population(domain[0], domain[1], inds_no)
    for i in xrange(0, iterations):
        population.crossover()
        population.mutate()
        population.selection(test_function)
        print '{}, {}'.format(i, population.avg)
        avg_x.append(i)
        avg_y.append(population.avg)


    results = {}
    for individual in population.population:
        # assign value of test function to individual value
        results[individual.float] = test_function(individual.float)

    the_best_x = max(results.iteritems(), key=operator.itemgetter(1))[0]
    the_best_y = test_function(the_best_x)
    print 'x:{:.3f} y:{:.3f}'.format(the_best_x, the_best_y)
    return the_best_x, the_best_y

algorithm_result = main(DOMAIN, ITERATION_NO, test_function, POPULATION_COUNT)


# draw function
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

x = np.linspace(0.5, 2.5, 500)
y = ((math.e**x) * np.sin(10*np.pi*x) + 1) / x

plt.plot(x, y, '-g')
plt.axvline(
    x=algorithm_result[0],
    ymin=0,
    ymax=algorithm_result[1],
    linewidth=3,
)
plt.text(algorithm_result[0] - 1, 4, 'Max f({:.3f}) = {:.3f}'.format(algorithm_result[0], algorithm_result[1], ), )
plt.xlabel('x', fontdict=font)
plt.ylabel('y', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
pass
# draw avg results
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

plt.plot(avg_x, avg_y, '-g')

plt.xlabel('iteration', fontdict=font)
plt.ylabel('avg result of target function', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
pass