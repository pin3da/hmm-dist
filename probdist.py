'''
A Probabilistic Distance Measure for Hidden
            Markov Models

AT&T Technical Journal
Vol. 64, No.2, February 1985
Printed in U.S.A.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def get_index(arr):
    cum = np.cumsum(arr)
    val = np.random.rand()
    return np.searchsorted(cum, val)

def fast_pow(B, e):
    ans = np.eye(B.shape[0])
    while (e > 0):
        if (e & 1):
            ans = ans.dot(B)
        B = B.dot(B)
        e >>= 1
    return ans

class HMM:
    def __init__(self):
        self.A = np.array([])
        self.B = np.array([])
        self.u = np.array([])
        self.a = np.array([])
        self.N = 0
        self.M = 0

    def read(self):
        self.N, self.M = [int(i) for i in input().split()]
        self.A = np.empty([self.N, self.N])
        self.B = np.empty([self.N, self.M])
        for i in range(self.N):
            tmp = [float(i) for i in input().split()]
            for j in range(self.N):
                self.A[i][j] = tmp[j]
        for i in range(self.N):
            tmp = [float(i) for i in input().split()]
            for j in range(self.M):
                self.B[i][j] = tmp[j]
        tmp = [float(i) for i in input().split()]
        self.u = np.array(tmp)

    def show(self):
        print (self.N, self.M)
        print (self.A)
        print (self.B)
        print (self.u)

    def gen_rw(self, T):
        cur_state = get_index(self.a)
        rw = [[cur_state, get_index(self.B[cur_state])]]
        for i in range(T - 1):
            cur_state = get_index(self.A[cur_state])
            rw.append([cur_state, get_index(self.B[cur_state])])
        return rw

    def find_st_dist(self):
        M = fast_pow(self.A, 10 ** 18)
        self.a = self.u.dot(M)
        self.a /= np.sum(self.a)
        b = self.a.dot(self.A)
        assert(np.allclose(self.a, b))

    def mu(self, path):
        '''path: observed sequence'''
        T = len(path)
        dp = np.zeros([T, self.N])

        for i in range(self.N):
            dp[T - 1][i] = self.B[i][path[T - 1]]

        for j in range(T - 2, -1, -1):
            for i in range(self.N):
                for k in range(self.N):
                    dp[j][i] += dp[j + 1][k] * self.A[i][k]
                dp[j][i] *= self.B[i][path[j]]

        ans = 0
        for i in range(self.N):
            ans += dp[0][i] * self.a[i]
        return ans

    def H(self, path):
        T = len(path)
        return (1.0 / T) * np.log(self.mu(path))

class Prob_dist:
    def __init__(self):
        self.M1 = HMM()
        self.M2 = HMM()

    def read_models(self):
        self.M1.read()
        self.M2.read()

    def find_st_distribution(self):
        self.M1.find_st_dist();
        self.M2.find_st_dist();

    def dist(self, T):
        rw1 = self.M1.gen_rw(T)
        y = [i[0] for i in rw1]
        lp1 = np.log(self.M1.mu(y))
        lp2 = np.log(self.M2.mu(y))
        return (1.0 / T) * (lp1 - lp2), lp1, lp2

if __name__ == '__main__':

    import sys

    dist = Prob_dist()
    dist.read_models()
    dist.find_st_distribution()
    x = []
    y = []
    lp1 = []
    lp2 = []
    for i in range(10, 500):
        x.append(i)
        a = 0
        b = 0
        c = 0
        times = 30
        for j in range(times):
            ap, bp, cp = dist.dist(i)
            a += ap
            b += bp
            c += cp
        y.append(a / times)
        lp1.append(b / times)
        lp2.append(c / times)

    # plt.ion()
    if (len(sys.argv) > 1 and sys.argv[1] == 'plot'):
        plt.plot(x, y)
        plt.title('Distance vs # observationts', fontsize=14)
        plt.xlabel('number of observations, T', fontsize=14)
        plt.ylabel('distance', fontsize=14)
        plt.show()

        plt.plot(x, lp1, label = 'mu(Ot | lamda 0)')
        plt.plot(x, lp2, label = 'mu(Ot | lamba)')
        plt.xlabel('number of observations, T', fontsize=14)
        plt.ylabel('log probability', fontsize=14)
        plt.legend()
        plt.show()

    print ('mean of dist: ', np.mean(y))
    print ('variance of dist: ', np.var(y))
