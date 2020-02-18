from training_routine.routine_registry import EXAMPLE_ROUTINE
from training_routine.base_routine import BaseRoutine


@EXAMPLE_ROUTINE.register('MyRoutine')
class ExampleRoutine(BaseRoutine):
    def __init__(self, net, cuda_device, is_cuda):
        """Initialize the routine with the net and cuda args."""
        pass

    def test_routine(self, sample):
        """Defines the test routine for one sample of the dataset."""
        pass

    def train_routine(self, sample):
        """Defines the train routine for one sample of the dataset."""
        pass

    def validation_routine(self, sample):
        """Defines the validation routine for one sample of the dataset."""
        pass
