class CondMargin:
    def __init__(self, h_x, h_y):
        self._fit = False
        self.h_x = h_x
        self.h_y = h_y

    def KX(self, x):
        return 1.*(np.abs(x) <= self.h_x)
    
    def Ky(self, y):
        return 1.*(np.abs(y) <= self.h_y)

    def fit(self, y, X):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        def pdf(y_, x_):
            y_ = y_.reshape(1, -1)
            X_ = x_.reshape(1, -1)
            Sy = self.Ky(y - y_.T)
            SX = self.KX(X - x_.T)
            return np.average(Sy, axis=1, weights=SX)
        def cdf(y_, x_):
            y_ = y_.reshape(1, -1)
            x_ = x_.reshape(1, -1)
            Sy = 1.*(y - y_.T <= 0)
            SX = self.KX(X - x_.T)
            return np.average(Sy, axis=0, weights=SX)
        self._fit = True
        self.pdf = pdf
        self.cdf = cdf
    
    def evaluate_cdf(self, y, X):
        assert(len(y)==len(X))
        assert(self._fit is True)
        y = y.reshape(1, -1)
        X = X.reshape(1, -1)
        cdf_res = self.cdf(y, X)
        #self.rank = (np.argsort(cdf_res)+1)/len(y)
        return cdf_res

    def rank(self, y, X):
        rank = self.evaluate_cdf(y, X).argsort().argsort()
        return rank
    

    def evaluate_pdf(self, y, X):
        assert(len(y)==len(X))
        assert(self._fit is True)
        y = y.reshape(1, -1)
        X = X.reshape(1, -1)
        return self.pdf(y, X)