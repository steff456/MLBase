"""Interface base routine."""


class BaseRoutine():
    """Base class for all the routines that include test, train and val."""

    def __init__(self, net, cuda_device, is_cuda):
        """Initialize the routine with the net and cuda args."""
        raise NotImplementedError

    def test_routine(self, sample):
        """Defines the test routine for one sample of the dataset."""
        raise NotImplementedError

    def train_routine(self, sample):
        """Defines the train routine for one sample of the dataset."""
        raise NotImplementedError

    def validation_routine(self, sample):
        """Defines the validation routine for one sample of the dataset."""
        raise NotImplementedError
