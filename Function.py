
import numpy as np
import random


class TestFunction(object):

    def __init__(self, n: int):
        self.n = n
        self.f_k = []
        self.norm_g_k = []

    def function(self, x: np.ndarray):
        raise NotImplementedError

    def gradient(self, x: np.ndarray):
        raise NotImplementedError

# each x_i is a 784 array (28x28 img).

class h(TestFunction):
        def __init__(self, n: int, x:np.ndarray, y:np.ndarray):
                super().__init__(n)
                self.x = x
                self.y = y
                self.x_size = len(x)

        def pi2(self, beta:np.ndarray, beta_0, i:int):
            b = beta[1:]
            exp = np.exp(- np.dot(self.x[i], b) - beta_0)
            return 1/(1 + exp)

        def pi(self, beta:np.ndarray, beta_0):
                b = beta[1:]
                # exp = np.exp(- np.dot(self.x[i], b) - beta_0)
                # return 1/(1 + exp)
                # new implementation
                pi = np.zeros(self.x_size)
                for i in range(self.x_size):
                    exp = np.exp(- np.dot(self.x[i], b) - beta_0)
                    pi[i] = 1/(1 + exp)
                return pi


        def function(self, beta: np.ndarray):
                beta_0 = beta[0]
                # sum = 0
                # for i in range(self.x_size):
                #        pi = self.pi(beta, beta_0, i)
                #        sum+=  self.y[i] * np.log(pi) + (1-self.y[i]) * np.log(1-pi)
                pi = self.pi(beta, beta_0)
                sum = np.sum(np.multiply(self.y, np.log(pi)) + np.multiply((1-self.y), np.log(1-pi)))
                return sum

        def gradient(self, beta: np.ndarray):
                beta_0 = beta[0]
                grad = np.zeros(self.n)
                pi = self.pi(beta, beta_0)
                y_p = self.y - pi

                grad[0] = np.sum(y_p)

                for k in range(self.n - 1):
                        grad[k+1] = np.dot(y_p, self.x[:,k])
                # sum = 0
                # for i in range(self.x_size):
                #     sum+= self.y[i] - self.pi(beta, beta_0, i)
                # grad[0] = sum
                #
                # for k in range(1, self.n):
                #     sum = 0
                #     for i in range(self.x_size):
                #         sum += (self.y[i] - self.pi(beta, beta_0, i)) * self.x[i][k-1]
                #
                #     grad[k] = sum
                return grad

