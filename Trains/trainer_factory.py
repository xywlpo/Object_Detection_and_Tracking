from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Trains.det_trainer import DetTrainer

trainer_factory = {
    'detection': DetTrainer
}