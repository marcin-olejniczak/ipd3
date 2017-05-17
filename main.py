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
        population_count = population_count

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
        pass

    def crossover(self, cross_propability=0.2):
        """
        Crossover individuals in population
        :return:
        """
        pass

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
                    return key

        choices = {}
        for individual in self.population:
            # assign value of test function to individual value
            choices[individual] = func(individual)

        new_population = []
        for n in self.population_count:
            new_population.append(
                weighted_random_choice(choices),
            )

        self.population = new_population

DOMAIN = (0.5, 2.5)

population = Population(0.5, 2.5, 100)

