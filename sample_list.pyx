from libc.stdlib cimport rand

def sample_list(array, int num_samples):
    samples = set()
    while True:
        r = rand() % len(array)
        samples.add(array[r])
        if len(samples) >= num_samples:
            return list(samples)

