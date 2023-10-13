
class IncorrectWeightsShapeError(Exception):

    def __init__(self, shape, n1, n2):
        """
        n1 and n2 target number of weights
        """
        self.message = ('CardiacTissue weigths {} is incorrect. '.format(shape) +
                        'Shape should be {} or {}'.format((*shape[:-1], n1),
                                                          (*shape[:-1], n2)))
