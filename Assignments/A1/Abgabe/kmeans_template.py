"""
K-means clustering.

NOTE: Please read README.txt

"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def analyze_kmeans():
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    X = np.genfromtxt("digit.txt")
    y = np.genfromtxt("labels.txt", dtype=int)
    distortions = []
    errs = []
    ks = range(1, 11)
    for k in ks:
        print("+++++++++++ LOOP NUMBER: ", k, "+++++++++++")
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
    #"""


def analyze_one_k(X, y, k):
    """
    Run the k-means analysis for a single value of k. Return the distortion and
    the mistake rate.
    """
    print( "Running k-means with k={0}".format(k) )
    clust = cluster(X, y, k)
    print( "Computing classification error." )
    err = compute_mistake_rate(y, clust)
    return clust["distortion"], err


def cluster(X, y, k, n_starts=5):
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
    def loop(X, i):
        """
        A single run of clustering.
        """
        Mu = initialize(X, k)
        N = X.shape[0]
        z = np.repeat(-1, N)        # So that initially all assignments change.
        while True:
            old_z = z
            z = assign(X, Mu)       # The vector of assignments z.
            Mu = update(X, z, k)    # Update the centroids
            if np.all(z == old_z):
                distortion = compute_distortion(X, Mu, z)
                return dict(Mu=Mu, z=z, distortion=distortion)

    # Main function body
    print( "Performing clustering." )
    results = [loop(X, i) for i in range(n_starts)]
    best = min(results, key=lambda entry: entry["distortion"])
    best["digits"] = label_clusters(y, k, best["z"])
    return best


def assign(X, Mu):
    """
    Assign each entry to the closest centroid. Return an Nx1 vector of
    assignments z.
    X is the NxD matrix of inputs.
    Mu is the kxD matrix of cluster centroids.
    """
    # done TODO: Compute the assignments z.
    # Initialize array for storing the assignment indices
    z = np.arange((len(X))).reshape(len(X), 1)
    # Loop over all observations to find the nearest
    for i in range(len(X)):
        # set initial distance high for later comparison
        dist = 10000000
        # Loop over all Centroids to find the nearest
        for j in range(len(Mu)):    
            # Utilize function from python special funtions package
            eu_dist = sp.spatial.distance.euclidean(X[i],Mu[j])
            # if distance is smaller then previous distance assign calculated distance as new smalles distance
            if ( eu_dist < dist ):
                # Coresponding index
                index = j
                dist = eu_dist
            else:
                continue
            # Store correct index in assignment array
        z = np.append(z,index)
    # Remove the initialization numbers
    z = z[1000:]
    return z


def update(X, z, k):
    """
    Update the cluster centroids given the new assignments. Return a kxD matrix
    of cluster centroids Mu.
    X is the NxD inputs as always.
    z is the Nx1 vector of cluster assignments.
    k is the number of clusters.
    """
    # done TODO: Compute the cluster centroids Mu.
    
    # Initialize np array as Mu (1xD matrix)
    Mu = np.zeros((k*len(X[0]))).reshape(k, len(X[0]))
    # Loop over all Mu to recalculate the new mean values
    for i in range(k):
        # Initialize initial sum as 0
        sum = 0
        # Loop over all z values to test in which group they are
        for j in range(len(z)):
            # if they are in the correct group, just sum them, increase the counter by 1 and continues 
            if ( i == z[j] ):
                sum += 1
                Mu[i] = Mu[i] + X[j]
            else:
                continue
        # After finishing the inner loop, calculate mean value
        Mu[i] /= sum
    return Mu


def compute_distortion(X, Mu, z):
    """
    Compute the distortion (i.e. within-group sum of squares) implied by NxD
    data X, kxD centroids Mu, and Nx1 assignments z.
    """
    # done TODO: Compute the within-group sum of squares (the distortion).
    # Loop over all centroids
    for i in range(len(Mu)):
        # Compare with all elements in X to get the correct assignments
        distortion = 0
        for j in range(len(z)):
            if ( z[j] == i ):
                # Calculate the sum of the vectors for calculating the new "center of weight"
                distortion = np.square( sp.spatial.distance.euclidean(X[j],Mu[i]) )
    return distortion


def initialize(X, k):
    """
    Randomly initialize the kxD matrix of cluster centroids Mu. Do this by
    choosing k data points randomly from the data set X.
    """
    # done TODO: Initialize Mu.
    # Initialize np array as Mu (1xD matrix)
    Mu = np.arange((len(X[0]))).reshape(1, len(X[0]))
    # Fill matrix
    for i in range(k):
        Mu = np.append(Mu, [X[np.random.randint(1,len(X))]], axis=0)
    # Remove first leftover item from initialization 
    Mu = np.delete(Mu,0,axis=0)
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
    # done TODO: Compute the cluster labelings.
    
    # Initialize an zero array for the labels
    labels = np.zeros((k)).reshape(k,)
    # loop over all centroids
    for i in range(k):
        # Initialize zero array for storing the abbundance of labels
        sum = np.zeros((4)).reshape(4,)
        # Variables for storing the abbundance
        N_1 = 0
        N_3 = 0
        N_5 = 0
        N_7 = 0
        # Loop over all assignments
        for j in range(len(z)):
            # Only increase the value if correct label is found
            if ( z[j] == i ):
                if ( y[j] == 1 ):
                    sum[0] += 1
                if ( y[j] == 3 ):
                    sum[1] += 1
                if ( y[j] == 5 ):
                    sum[2] += 1
                if ( y[j] == 7 ):
                    sum[3] += 1
        # Store index with the maximal amount of events 
        index = np.argmax(sum)
        # Assign correct label to labels array
        if ( index == 0 ):
            labels[i] = 1
        if ( index == 1 ):
            labels[i] = 3
        if ( index == 2 ):
            labels[i] = 5
        if ( index == 3 ):
            labels[i] = 7
    print(labels)
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
    analyze_kmeans()

if __name__ == '__main__':
    main()
