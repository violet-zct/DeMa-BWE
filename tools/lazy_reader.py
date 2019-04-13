import gzip
import numpy as np

def lazy_file_reader(path):
    with gzip.open(path, 'rb') as fin:
        for line in fin:
            yield [int(i) for i in line.decode().strip().split()]


class lazy_file_iterator():
    def __init__(self, path):
        self.fin = gzip.open(path, "rb")
        self.buf_size = 100
        self.cache = []
        self.pointer = 0

    def reset(self):
        self.fin.seek(0)

    def __next__(self):
        if len(self.cache) <= 0:
            for _ in range(self.buf_size):
                line = self.fin.readline()
                if line == '':
                    self.reset()
                self.cache.append(line)

        return [int(i) for i in self.cache.pop().decode().strip().spit()]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.cache)

# No need to be lazy any more
class words_sampler_iterator():
    def __init__(self, probs, buffer_size, batch_size, uniform_sample):
        self.probs = probs
        self.set_size = len(probs)
        if uniform_sample:
            self.probs = np.ones(self.set_size) * 1. / self.set_size

        self.batch_size = batch_size
        self.cache = np.arange(buffer_size)

    def retrieve_cache(self):
        return self.cache

    def __next__(self):
        return np.random.choice(self.set_size, self.batch_size, replace=True, p=self.probs)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.cache)