from libc.stdlib cimport rand

def sample_list(array, int num_samples):
    samples = set()
    arr_len = len(array)
    while True:
        cdef int r = rand() % arr_len
        samples.add(array[r])
        if len(samples) >= num_samples:
            return list(samples)

