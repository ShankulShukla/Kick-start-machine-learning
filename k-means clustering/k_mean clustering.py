import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
class clustering:

    def __init__(self, k, n_iter, tol):
        self.k = k
        self.n_iteration = n_iter
        self.tolerance = tol

    def fit(self, X):
        #initializing the clusters
        index = np.random.randint(0, len(X)-1, self.k)
        self.centroid, assign_centroid = [], {}
        for i in range(self.k):
            self.centroid.append(X[index[i]])
            assign_centroid[i] = []

        for _ in range(self.n_iteration):

            # find centroid step
            for x in X:
                dist = np.linalg.norm(self.centroid - x, axis=1)
                c = np.argmin(dist)
                assign_centroid[c].append(x)

            # move centroid step
            prev_centroid = self.centroid
            self.centroid = []
            for i in range(self.k):
                self.centroid.append(np.mean(assign_centroid[i],axis=0))
            # checking for tolerance
            diff = 0.
            for i in range(self.k):
                diff += np.linalg.norm(prev_centroid[i] - self.centroid[i])
            if diff < self.tolerance:
                print('Early convergence!!!')
                break
            for i in range(self.k):
                assign_centroid[i] = []


    def predict(self, X):
        c = []
        for x in X:
            dist = np.linalg.norm(self.centroid - x, axis=1)
            c.append(np.argmin(dist))
        return c

X, y = make_blobs(n_samples=150, centers= 3, n_features= 2, random_state=20)
clf = clustering(3, 500, 0.0001)
plt.scatter(X[:,0],X[:,1])
clf.fit(X)
for c,i in enumerate(clf.centroid):
    plt.scatter(i[0],i[1],label='class {}'.format(c),marker='s')
plt.legend()
plt.show()
# multiple run of the code will show the need of effective initialization of cluster centroids