"""
K-means clustering.
"""

import numpy as np
from matplotlib import pyplot as plt


def analyze_kmeans(k):
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    X = np.genfromtxt("digit.txt")
    # Specification of the dimension
    N = len(X)
    D = len(X[0])
    y = np.genfromtxt("labels.txt", dtype=int)
    distortions = []
    errs = []
    ks = range(1, 11)
    ###
    test = 53
    Mu = initialize(X, k, N)
    print(Mu[0][test])
    for i in range(500):
        z = assign(X, Mu, N, D)
        Mu = update(X, z, k, D, Mu)
        print(Mu[0][test])
    #cluster(X, y, k, N, D, n_starts=5)
    ###
    """
    for k in ks:
        distortion, err = analyze_one_k(X, y, k)
        distortions.append(distortion)
        errs.append(err)
    fig, ax = plt.subplots(2, figsize=(8, 6))
    ax[0].plot(ks, distortions, marker=".")
    ax[0].set_ylabel("Distortion")
    ax[1].plot(ks, errs, marker=".")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Mistake rate")
    ax[0].set_title("k-means performance")
    fig.savefig("kmeans.png")
    """


def analyze_one_k(X, y, k):
    """
    Run the k-means analysis for a single value of k. Return the distortion and
    the mistake rate.
    """
    #print "Running k-means with k={0}".format(k)
    clust = cluster(X, y, k)
    print( "Computing classification error." )
    err = compute_mistake_rate(y, clust)
    return clust["distortion"], err


def cluster(X, y, k, N, D, n_starts):
    """
    Run k-means a total of n_starts times. Returns the results from the run that
    had the lowest within-group sum of squares (i.e. the lowest distortion).

    Inputs
    ------
    X is an NxD matrix of inputs.
    y is a Dx1 vector of labels.
    n_starts says how many times to randomly re-initialize k-means. You don't
        need to change this.

    Outputs
    -------
    The output is a dictionary with the following fields:
    Mu is a kxD matrix of cluster centroids
    z is an Nx1 vector assigning points to clusters. So, for instance, if z[4] =
        2, then the algorithm has assigned the 4th data point to the second
        cluster.
    distortion is the within-group sum of squares, a number.
    """
    def loop(X, i, k, N, D):
        """
        A single run of clustering.
        """
        Mu = initialize(X, k, N)
        N = X.shape[0]
        z = np.repeat(-1, N)        # So that initially all assignments change.
        while True:
            old_z = z
            z = assign(X, Mu, N, D)       # The vector of assignments z.
            Mu = update(X, z, k, D, Mu)    # Update the centroids
            if np.all(z == old_z):
                distortion = compute_distortion(X, Mu, z)
                return dict(Mu=Mu, z=z, distortion=distortion)

    # Main function body
    print( "Performing clustering." )
    results = [loop(X, i, k, N, D) for i in range(n_starts)]
    best = min(results, key=lambda entry: entry["distortion"])
    best["digits"] = label_clusters(y, k, best["z"])
    return best


def assign(X, Mu, N, D):
    """
    Assign each entry to the closest centroid. Return an Nx1 vector of
    assignments z.
    X is the NxD matrix of inputs.
    Mu is the kxD matrix of cluster centroids.
    """
    # done TODO: Compute the assignments z.
    # Find position of centorids in X and store this information in a local array
    positions =[]
    # Loop over all centroids
    for i in range(len(Mu)):
        # Loop over all datapoints in X
        for j in range(N):
            # initialize variable to store information if row is equal or not
            is_equal = 0
            # Loop over all values in row j of matrix X to check if Mu[i] is equal to X[j] 
            for m in range(D):
                # Check if Mu[i] is equal to X[j] by testing all datapoints m  
                if ( X[j][m] == Mu[i][m] ):
                    is_equal = 1
                    continue
                else:
                    is_equal = 0
                    break
            # At the end of the day, the positions are j
            if ( is_equal == 1 ):
                positions.append(j)
                continue
            else:
                continue
    # Initialize Nx2 vector
    z = []
    # Find nearest centroid by looping over all datapoints
    for i in range(N):
        min_dist = N*10000000000
        dist = 0
        nearest = [0,0]
        # Compare all datapoints with every centorid 
        for j in range(len(Mu)):
            # Calculate Eucledean distance between datapoints 
            for m in range (D):
                dist += ( X[i][m] - Mu[j][m] )**2
            dist = np.sqrt(dist)
            #print(dist)
            if ( np.absolute(dist) < min_dist ):
                nearest = [j,positions[j]]
                min_dist = np.absolute(dist)
            else:
                continue
        # Feed nearest centorid in z arrey for index i
        z.append(nearest)
    #print(z)
    return z


def update(X, z, k, D, Mu):
    """
    Update the cluster centroids given the new assignments. Return a kxD matrix
    of cluster centroids Mu.
    X is the NxD inputs as always.
    z is the Nx1 vector of cluster assignments.
    k is the number of clusters.
    """
    # TODO: Compute the cluster centroids Mu.
    # First we have to delete entries of the Mu array
    for i in range(len(Mu)):
        for j in range(D):
            Mu[i][j] = 0
    sum_of_cluster = 0
    # Scan through all elements in Mu
    for i in range(len(Mu)):
        # Compare withh all elements in X
        for j in range(len(z)):
            if ( z[j][0] == i ):
                sum_of_cluster += 1
                # Calculate the sum of the vectors for calculating the new "center of weight"
                for m in range(D):
                    Mu[i][m] += X[j][m]
        for m in range(D):
            Mu[i][m] /= sum_of_cluster
        sum_of_cluster = 0
    #print(Mu)
    return Mu


def compute_distortion(X, Mu, z):
    """
    Compute the distortion (i.e. within-group sum of squares) implied by NxD
    data X, kxD centroids Mu, and Nx1 assignments z.
    """
    # TODO: Compute the within-group sum of squares (the distortion).
    distortion = None
    return distortion


def initialize(X, k, N):
    """
    Randomly initialize the kxD matrix of cluster centroids Mu. Do this by
    choosing k data points randomly from the data set X.
    """
    # done TODO: Initialize Mu.
    # Initialize np array as Mu (kxD matrix)
    Mu = []
    # Fill matrix
    for i in range(k):
        Mu.append(X[np.random.randint(1,N)])
    return Mu


def label_clusters(y, k, z):
    """
    Label each cluster with the digit that occurs most requently for points
    assigned to that cluster.
    Return a kx1 vector labels with the label for each cluster.
    For instance: if 20 points assigned to cluster 0 have label "3", and 40 have
    label "5", then labels[0] should be 5.

    y is the Nx1 vector of digit labels for the data X
    k is the number of clusters
    z is the Nx1 vector of cluster assignments.
    """
    # TODO: Compute the cluster labelings.
    labels = None
    return labels


def compute_mistake_rate(y, clust):
    """
    Compute the mistake rate as discussed in section 3.4 of the homework.
    y is the Nx1 vector of true labels.
    clust is the output of a run of clustering. Two fields are relevant:
    "digits" is a kx1 vector giving the majority label for each cluster
    "z" is an Nx1 vector of final cluster assignments.
    """
    def zero_one_loss(xs, ys):
        return sum(xs != ys) / float(len(xs))

    y_hat = clust["digits"][clust["z"]]
    return zero_one_loss(y, y_hat)


def main():
    analyze_kmeans(10)


if __name__ == '__main__':
    main()
