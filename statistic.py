
class TStatistic:
    """
    Class for Cramer Von Mises Statistic. Denoted by T_{1,n} in the paper. 
    This statistic is the L2-norm over t and u of L(u, Kt).
    Can be computed as Sum_{i,j} M(i,j)K(X_i - X_j) (see Sec. Computation of the test statistic).
    """

    def __init__(self, sigma_X, gamma):
        self.sigma_X = sigma_X
        self.gamma = gamma
        self.Kts = Kt_sigma(0, 2*self.sigma_X, kernel='RBF')
        self.mij = Mij(gamma=self.gamma)
    
    def evaluate(self, n_iter, y, X):
        T_H0s = []
        for i in tqdm(np.arange(n_iter)):
            y_, _, X_, _ = train_test_split(y, X, test_size=.25)
            T_H0s.append(self.evaluate_(y_, X_))
        return T_H0s
    
    def evaluate_(self, y, X):
        T_1_n = 0
        self.mij.fit(y, X)
        for (i, xi) in enumerate(X):
            for (j, xj) in enumerate(X):
                T_1_n += self.Kts.evaluate(xi - xj)*self.mij.evaluate(i, j)
        return T_1_n/len(X)**2

