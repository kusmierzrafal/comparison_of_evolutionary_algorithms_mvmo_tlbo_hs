from numpy import vstack as numpy_vstack


def vstack(gen_ind: callable, pop_size: int):
    """
    gen_ind has to be function returning individual from population
    """
    ind = gen_ind()
    return numpy_vstack([ind] * pop_size)
