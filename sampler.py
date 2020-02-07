class CondIndepSampler:
    def __init__(self, sigma_X, kernel='constant'):
        self.sigma_X = sigma_X
        self.kernel = kernel
        self.fit_ = False

    def fit(self, X, y):
        self.X_ = deepcopy(X)
        self.y_ = deepcopy(y)
        self.size_X = len(self.X_)
        self.fit_ = True

    def sample_wrapper(self):
        if not self.fit_:
            raise NotImplementedError
        else:
            sampled_X = np.random.randint(0, len(self.X_),)
            sampled_X =  self.X_.iloc[sampled_X]
            ktxi = Kt_sigma(sampled_X, sigma=self.sigma_X, kernel=self.kernel)
            nb_xi = ktxi.evaluate(self.X_)
            nb_xi /= nb_xi.sum()
            sampled_y_given_X = np.empty(self.y_.shape[1])
            for i in range(self.y_.shape[1]):
                sampled_y_given_X[i] = self.y_.iloc[np.random.choice(np.arange(self.size_X), p=nb_xi), i]
            return np.array([sampled_X, *sampled_y_given_X])

    def sample(self, size_sample):
        return np.array([self.sample_wrapper() for i in range(size_sample)])