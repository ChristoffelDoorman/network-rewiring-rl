import sys
from pathlib import Path

sys.path.append('/relnet')

from relnet.objective_functions.objective_functions import *


def get_metrics():
    metrics = {'MERW':
                   {'fn': MERW(),
                    'kwargs': {}
                    },
               'Shannon':
                   {'fn': GlobalEntropy(),
                    'kwargs': {}
                    },
               'BZ2':
                   {'fn': BZ2CompressRatio(),
                    'kwargs': {'num_shuffles': 10}
                    },
               'BDM':
                   {'fn': BDM(),
                    'kwargs': {'num_shuffles': 10}
                    },
               'local_metrics':
                   {'fn': LocalMetric(),
                    'kwargs': {'n_subg': 8,
                               'num_shuffles': 10,
                               'metrics': ['local_MERW', 'local_Shannon', 'local_BZ2', 'local_BDM']}
                    }
               }

    return metrics
