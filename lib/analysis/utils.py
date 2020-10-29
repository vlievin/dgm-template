class RunningMean():
    def __init__(self):
        self.mean = None
        self.n = 0

    def update(self, x, k=1):
        """use k > 1 if x is averaged over `k` points, k > 1.
        Useful when averaging over mini-batch with different dimensions."""
        if self.mean is None:
            self.mean = x
        else:
            self.mean = self.n / (self.n + k) * self.mean + k / (self.n + k) * x

        self.n += k

    def __call__(self):
        return self.mean


class RunningVariance():
    def __init__(self):
        self.n = 0
        self.Ex = None
        self.Ex2 = None
        self.K = None

    def update(self, x):
        self.n += 1
        if self.K is None:
            self.K = x
            self.Ex = x - self.K
            self.Ex2 = (x - self.K) ** 2
        else:
            self.Ex += x - self.K
            self.Ex2 += (x - self.K) ** 2

    def __call__(self):
        return (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1)
