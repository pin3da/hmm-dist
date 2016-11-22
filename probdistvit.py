'''
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.4174&rep=rep1&type=pdf
eq 6
'''

from probdist import HMM
import numpy as np


eps = 1e-15

class Prob_dist:
    def __init__(self):
        self.M1 = HMM()
        self.M2 = HMM()

    def read_models(self):
        self.M1.read()
        self.M2.read()

    def find_st_distribution(self):
        self.M1.find_st_dist()
        self.M2.find_st_dist()

    def dec(self):
        ans = 0.0
        for i in range(self.M1.N):
            tmp = self.M1.B[i].dot(self.M2.B[i])
            ans += tmp
        ans /= self.M1.N
        return np.sqrt(ans)

    def dist(self):
        ans = 0.0
        for i in range(self.M1.N):
            for j in range(self.M1.N):
                tmp = np.log(self.M2.A[i][j]+eps) - np.log(self.M1.A[i][j]+eps)
                ans += self.M1.A[i][j] * self.M1.a[i] * tmp
            for k in range(self.M1.M):
                tmp = np.log(self.M2.B[i][k]+eps) - np.log(self.M1.B[i][k]+eps)
                ans += self.M1.B[i][k] * self.M1.a[i] * tmp
        return ans


if (__name__ == '__main__'):
    dist = Prob_dist()
    dist.read_models()
    dist.find_st_distribution()
    print ('Dist euc', dist.dec())
    print ('Dist vit', dist.dist())
