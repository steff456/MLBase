"""Complete arguments for the program."""
import argparse


def get_args():
    """Get the arguments for the training - testing of the net."""
    parser = argparse.ArgumentParser(
        description='Petrography training routine')

    # Dataloading-related settings
    parser.add_argument('--config', default='experiments/example.yaml',
                        help='location for the config file')

    return parser.parse_args()
