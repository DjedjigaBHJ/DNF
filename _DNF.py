import math
import numpy as np
import matplotlib.pyplot as plt
import random

class MyDNF:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.dt = 0.7
        self.to = 0.64

        self.C_exc = 1.25
        self.Sigma_exc = 3
        self.C_inh = 0.7
        self.Sigma_inh = 15

        self.Input = np.zeros([width, height])
        self.mat_noeurone = np.zeros([width, height])
        self.tmp_mat = self.mat_noeurone
        self.sum_U_W_mat = np.zeros([width, height])
        self.direction1 = 1
        self.direction2 = 1
        self.a = [3, 3]
        self.b = [18, 18]
        self.sigma = 2
        self.vitesse =1

    def euclidean_dist(self,x, y):
        res = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        return res

    def gaussian(self,a, x, sigma):
        res = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (self.euclidean_dist(a, x) ** 2 / sigma ** 2))
        return res

    def scale(self,X, x_min, x_max):
        nom = (X - X.min()) * (x_max - x_min)
        denom = X.max() - X.min()
        denom = denom + (denom is 0)
        return x_min + nom / denom

    def aleaGauss(self,sigma):
        U1 = random.random()
        U2 = random.random()
        return sigma * math.sqrt(-2 * math.log(U1)) * math.cos(2 * math.pi * U2)

    def noise(self):
        ### Ajout d'un bruit gaussien
        for i in range(self.width):
            for j in range(self.height):
                self.Input[i, j] = self.Input[i, j] + (self.aleaGauss(0.1) /2)

    def vitesseGauss(self,a,b, vitesse, sigma):

        newA = a
        newB = b

        if self.direction1 == 1:
            newA[0] = a[0] + vitesse
        if self.direction1 == 2:
            newA[1] = a[1] + vitesse
        if self.direction1 == 3:
            newA[0] = a[0] - vitesse
        if self.direction1 == 4:
            newA[1] = a[1] - vitesse

        if newA[0] > (self.width - 2*self.sigma) and self.direction1 == 1:
            self.direction1 = 2
            newA[0] = a[0]
        if newA[0] <  2*self.sigma and self.direction1 == 3:
            self.direction1 = 4
            newA[0] = a[0]
        if newA[1] > (self.height - 2*self.sigma) and self.direction1 == 2:
            self.direction1 = 3
            newA[1] = a[1]
        if newA[1] <  2*self.sigma and self.direction1 == 4:
            self.direction1 = 1
            newA[1] = a[1]

        if self.direction2 == 1:
            newB[0] = b[0] + vitesse
        if self.direction2 == 2:
            newB[1] = b[1] + vitesse
        if self.direction2 == 3:
            newB[0] = b[0] - vitesse
        if self.direction2 == 4:
            newB[1] = b[1] - vitesse

        if newB[0] > (self.width- 2*self.sigma) and self.direction2 == 1:
            self.direction2 = 2
            newB[0] = b[0]
        if newA[0] <  2*self.sigma  and self.direction2 == 3:
            self.direction2 = 4
            newB[0] = b[0]
        if newA[1] > (self.height - 2*self.sigma) and self.direction2 == 2:
            self.direction2 = 3
            newB[1] = b[1]
        if newB[1] < 2*self.sigma and self.direction2 == 4:
            self.direction2 = 1
            newB[1] = b[1]

        self.Input = self.gaussian_activity(newA, newB, sigma)
        self.noise()
        self.a = newA
        self.b = newB

    def gaussian_activity(self, a, b, sigma):

        matr1 = np.zeros((self.width, self.height))
        matr2 = np.zeros((self.width, self.height))
        matRes = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                matr1[i, j] = self.gaussian(a, [i, j], sigma)
                matr2[i, j] = self.gaussian(b, [i, j], sigma)
                matRes[i, j] = matr1[i, j] + matr2[i, j]
        matRes = self.scale(matRes, 0, 1)
        return matRes

    def difference_of_gaussian(self, distance ):
        w = self.C_exc * math.exp(-(distance ** 2) / (2 * self.Sigma_exc ** 2)) - self.C_inh * math.exp(
            -(distance ** 2) / (2 * self.Sigma_inh ** 2))
        return w

    def sum_u_w(self, x):
        sum = 0
        for i in range(self.width):
            for j in range(self.height):
                sum = sum + self.tmp_mat[i, j] * self.difference_of_gaussian(self.euclidean_dist((x[0], x[1]), (i, j)))
        return sum

    def sum_U_W(self):
        for i in range(self.width):
            for j in range(self.height):
                self.sum_U_W_mat[i, j] = self.sum_u_w([i, j])


    def update_neuron(self, x):  # x est un vecteur 2D entier

        dis = self.mat_noeurone[x[0]][x[1]] + (self.dt / self.to) * (-self.mat_noeurone[x[0], x[1]] +
                                                                     self.sum_U_W_mat[x[0], x[1]] + self.Input[x[0]][x[1]])
        if dis < 0:
            self.mat_noeurone[x[0]][x[1]] = 0
        elif dis > 1:
            self.mat_noeurone[x[0]][x[1]] = 1
        else:
            self.mat_noeurone[x[0]][x[1]] = dis

    def synchronous_run(self):
        self.Input = self.gaussian_activity(self.a, self.b, self.sigma)
        self.noise()
        fig = plt.figure()

        for k in range(25):
            print(k)
            self.tmp_mat = self.mat_noeurone
            self.sum_U_W()
            for i in range(self.width):
                for j in range(self.height):
                    self.update_neuron([i, j])
            plt.pause(1)
            plt.subplot(121)
            plt.imshow(self.Input, cmap='hot', interpolation='nearest', animated=True)
            plt.subplot(122)
            plt.imshow(self.mat_noeurone, cmap='hot', interpolation='nearest', animated=True)
            self.vitesseGauss(self.a, self.b, self.vitesse, self.sigma)

    def asynchronous_run(self):
        self.Input = self.gaussian_activity([3, 3], [10, 15], 0.2)
        self.noise()
        fig = plt.figure()
        for k in range(25):
            print(k)
            self.tmp_mat = self.mat_noeurone
            self.sum_U_W()
            for i in range(self.width):
                for j in range(self.height):
                    al = random.random()
                    if al > 0.5:
                        self.update_neuron([i, j])
            plt.pause(1)
            plt.subplot(121)
            plt.imshow(self.Input, cmap='hot', interpolation='nearest', animated=True)
            plt.subplot(122)
            plt.imshow(self.mat_noeurone, cmap='hot', interpolation='nearest', animated=True)
            self.vitesseGauss(self.a, self.b, self.vitesse, self.sigma)


def rKN(x, fx, n, hs):
        # fx(x) ==> self.mat_noeurone[x[0]][x[1]]
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        xk = []
        for i in range(n):
            k1.append(fx[i](x) * hs)
        for i in range(n):
            xk.append(x[i] + k1[i] * 0.5)
        for i in range(n):
            k2.append(fx[i](xk) * hs)
        for i in range(n):
            xk[i] = x[i] + k2[i] * 0.5
        for i in range(n):
            k3.append(fx[i](xk) * hs)
        for i in range(n):
            xk[i] = x[i] + k3[i]
        for i in range(n):
            k4.append(fx[i](xk) * hs)
        for i in range(n):
            x[i] = x[i] + (k1[i] + 2 * (k2[i] + k3[i]) + k4[i]) / 6
        return x


if __name__ == '__main__':
    dnf_test = MyDNF(25, 25)
    dnf_test.synchronous_run()

