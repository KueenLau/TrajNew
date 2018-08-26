import numpy as np
from numpy.linalg import cholesky
from metric_learn import LMNN
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mu = np.array([[1, 5]])
    Sigma = np.array([[1.5, 0.5], [1.5, 3]])
    R = cholesky(Sigma)
    s = np.dot(np.random.randn(100, 2), R) + mu
    label = np.zeros((100,1))

    mu1 = np.array([[5,10]])
    Sigma1 = np.array([[1, 0.5], [1.5, 3]])
    R1 = cholesky(Sigma1)
    s1 = np.dot(np.random.randn(100, 2), R1) + mu1
    label1 = np.zeros((100,1)) + 1

    plt.subplot(121)
    plt.plot(s[:,0],s[:,1],".",color='red')
    plt.plot(s1[:,0],s1[:,1],".",color='blue')

    l1 = list(label)
    l2 = list(label1)
    l1.extend(l2)
    labels = np.array(l1)

    s_ = np.vstack((s, s1))
    print(s_.shape)
    print(labels.shape)

    lmnn = LMNN(k=2,min_iter=500,learn_rate=1e-6)
    lmnn.fit(s_, labels)
    s_new = lmnn.transform(s_)
    plt.subplot(122)
    plt.plot(s_new[:,0], s_new[:,1], ".")
    plt.show()