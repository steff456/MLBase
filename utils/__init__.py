# -*- coding: utf-8 -*-

"""Misc data and other helping utilities."""


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize Average Meter class."""
        self.reset()

    def reset(self):
        """Reset method."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update method."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
