from evolutionary_algorithms.helpers import vstack
from evolutionary_algorithms.population import Population


class Crossover:

    def __init__(
        self,
        crossover_type,
    ):

        crossover_type_dict = {"mapping_transformation": self.init_mapping_crossover()}
        self.cross = crossover_type_dict[crossover_type]

    def init_mapping_crossover(self):

        return self.mapping_crossover

    @staticmethod
    def mapping_crossover(population: Population, mask):
        """
        As input gets population to crossover and mask telling which individuals to cross
        """
        population_transposed = population.population.T
        pop_size = population.get_size()
        best_ind_pop_size = vstack(population.get_best_archive_individual, pop_size)
        population_transposed[mask] = best_ind_pop_size[mask]
