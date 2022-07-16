import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def build_noise_schedule(noise_schedule):
    if isinstance(noise_schedule, list):
        return np.array(noise_schedule, dtype=np.float32)
    elif isinstance(noise_schedule, np.ndarray):
        return noise_schedule.astype(np.float32)
    elif isinstance(noise_schedule):
        return np.linspace(
            start=noise_schedule.start,
            stop=noise_schedule.stop,
            num=noise_schedule.num)
    else:
        raise ValueError('Unsupported type.')