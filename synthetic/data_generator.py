import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

log = logging.getLogger("slda")
log.addHandler(logging.StreamHandler())


class DataGenerator():
    def __init__(self, D = 250, W = 1000, K = 10, sigma2=0.005, Nl = 10, seed=42, **kwargs):
        """
        :param D: number of documents
        :param W: number of words
        :param K: number of topics
        :param sigma2: keyword variability
        :param Nl: number of layers
        :param alpha: topic-word prior. default 1/K
        :param beta: doc-topic prior. default 1/K
        :param eta: normal mean to sample keywords. default 10/K
        :param seed: random seed
        """
        self.D = D  # documents
        self.W = W  # words
        self.K = K  # 10 topics
        self.alpha = 1./K
        self.beta = 1./K
        self.eta = 10./K
        self.sigma2 = sigma2  # 0.005
        self.Nl = Nl #Number of layers
        self.Nd = [100 for doc in range(D)]
        self.rng = np.random.RandomState(seed=seed)
        self._cache = {}

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]
        if "beta" in kwargs.keys():
            self.beta = kwargs["beta"]
        if "eta" in kwargs.keys():
            self.eta = kwargs["eta"]
        if "Nd" in kwargs.keys():
            self.Nd = kwargs["Nd"]

    def _is_cached(self, key:str):
        return key in self._cache.keys()

    def _get_from_cache(self, key:str):
        return self._cache[key]
    
    def _put_in_cache(self, key: str, val)->None:
        self._cache[key] = val

    @property
    def phi(self):
        """
        A vector shaped (K, W) containing word distributions of topic k
        """
        if self._is_cached("phi"):
            return self._get_from_cache("phi")

        phi = self.rng.dirichlet(np.repeat(self.alpha, self.W), size=self.K)
        assert(phi.shape == (self.K, self.W))
        self._put_in_cache("phi", phi)
        return phi

    @property
    def theta(self):
        """
        A vector shaped (D, K) containing topic distributions of document d
        """
        if self._is_cached("theta"):
            return self._get_from_cache("theta")

        theta = self.rng.dirichlet(np.repeat(self.beta, self.K), size=self.D)
        assert(theta.shape == (self.D, self.K))
        self._put_in_cache("theta", theta)
        return theta
    
    @property
    def z(self):
        """
        A vector shaped (D, W) containing topic assignment of  word w in document d
        """
        if self._is_cached("z"):
            return self._get_from_cache("z")
        
        z = np.array([[np.argmax(self.rng.multinomial(1, self.theta[i])) for j in range(self.Nd[i])]
                    for i in range(self.D)])
        self._put_in_cache("z", z)
        return z
    
    @property
    def z_bar(self):
        """
        A vector shaped (D), average of Z over words
        """
        if self._is_cached("z_bar"):
            return self._get_from_cache("z_bar")

        z_bar = np.average(self.z, 1)
        self._put_in_cache("z_bar", z_bar)
        return z_bar

    @property
    def Ys(self):
        """
        A vector shaped (L, D, W) containing word assignment of document d in layer l
        """

        if self._is_cached("Ys"):
            return self._get_from_cache("Ys")
        Ys = [[[self.rng.normal(self.eta*z_bar_d, self.sigma2) for _ in range(self.Nd[d])]
            for d, z_bar_d in enumerate(self.z_bar)] for _ in range(self.Nl)]
        
        self._put_in_cache("Ys", Ys)
            
        return Ys

    @property
    def Ys_random(self):
        """
        A vector shaped (L, D, W) containing word assignment of document d in layer l
        """
        if self._is_cached("Ys_random"):
            return self._get_from_cache("Ys_random")
        Ys = [[self.rng.uniform(0, 50, size=self.Nd[d])
               for d, z_bar_d in enumerate(self.z_bar)] for _ in range(self.Nl)]
        self._put_in_cache("Ys_random", Ys)
        return Ys

    @property
    def Y(self):
        """
        A vector shaped (D, W) containing keyword assignment of document d
        """
        if self._is_cached("Y"):
            return self._get_from_cache("Y")

        Y = [self.rng.normal(self.eta*z_bar_d, self.sigma2) for z_bar_d in self.z_bar]
        assert(len(Y) == self.D)

        self._put_in_cache("Y", Y)
        return Y

    @property
    def Wij(self):
        if self._is_cached("Wij"):
            return self._get_from_cache("Wij")
        W = np.array([[np.argmax(self.rng.multinomial(1, self.phi[self.z[i, j]])) for j in range(self.Nd[i])]
                    for i in range(self.D)])

        self._put_in_cache("Wij", W)
        return W

    @property
    def gamma(self):
        """
        Round Y
        """
        if self._is_cached("gamma"):
            return self._get_from_cache("gamma")

        gamma = np.array([int(y) for y in self.Y])
        self._put_in_cache("gamma", gamma)
        return gamma

    def get_BoW(self)->pd.DataFrame: 
        """
        :return: Bag of Words
        """       
        df = pd.DataFrame()
        for doc in range(self.D):
            doc_dict, doc_ab = np.unique(self.Wij[doc], return_counts=True)
            df = df.join(pd.Series(index=doc_dict, data=doc_ab,
                        name="doc_{}".format(doc)), how="outer")
        df = df.fillna(0).astype(int)
        return df

    def get_BoT(self):
        """
        Return Bag of Token
        """
        df_meta = pd.DataFrame()
        for doc in range(self.D):
            doc_dict, doc_ab = np.unique(self.gamma[doc], return_counts=True)
            df_meta = df_meta.join(
                pd.Series(index=doc_dict, data=doc_ab, name="doc_{}".format(doc)), how="outer")
        df_meta = df_meta.fillna(0).astype(int)
        return df_meta


    def get_abundances(self)->np.array:
        """
        Abundance a.k.a the number of times a word appear in the whole corpus
        """
        if self._is_cached("a"):
            return self._get_from_cache("a")
        a = np.array([int(y) for y in self.Y])
        
        self._put_in_cache("a", a)
        return a

if __name__=="__main__":
    generator = DataGenerator(a=3)
    print(generator.get_BoW().head())
    print(generator.get_BoT().head())
