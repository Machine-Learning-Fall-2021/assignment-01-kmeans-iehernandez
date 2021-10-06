import numpy as np
import random
import math

class KMeans():
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None # numpy array # see create_random_centroids for more info
        self.labels_ = None # predictions # numpy array of size len(input)

    def fit(self, input: np.ndarray) -> np.array: 
        """
            Fitting a model means to train the model on some data using your specific algorithm. 
            Typically, you also provide a set of a labels along with your input.
            However, this is an unsupervised algorithm so we don't have y (or labels) to consider! 
                If you're not convinced, look up any supervised learning algorithm on sklearn: https://scikit-learn.org/stable/supervised_learning.html
                If you can explain the difference between the fit function of this unsupervised algorithm and any other supervised algorith, you get 5 extra credit points towards this assignment. 
            This function will simply return the cluster centers, but it will also update your cluster centers and predictions.
        """

        # initialize cluster centers by selecting random points from input data
        self.cluster_centers_= np.array(random.sample(list(input), self.n_clusters))
        
        # initialize the labels
        self.labels_ = [0 for l in range(len(input))]
        
        # keep track of cluster movement
        previous = np.array([[] for l in range(self.n_clusters)])
        
        for i in range(self.max_iter): # iterate until it reaches max iterations
            # holds point of centroids of the previous iteration to detect centroid movement
            current_points_to_centroids = [[] for l in range(self.n_clusters)]
            
            for p in range(len(input)): # iterate through all points
                centroid_distance = [] 
                
                for j in range(self.n_clusters):  # min distance to find closest centroid
                    centroid_distance.append(self.calculate_distance(input[p], self.cluster_centers_[j]))
                    
                min_value = min(centroid_distance)
                min_index = centroid_distance.index(min_value) # index of closest centroid
    
                current_points_to_centroids[min_index].append(input[p]) # associate points to centroids
                self.labels_[p] = min_index
            
            if (np.array_equal(previous, self.cluster_centers_)):  # if no centroid movement return    
                self.labels_ = np.array(self.labels_)
                return self.cluster_centers_
            else:
                previous = self.cluster_centers_  # update previous cluster centers
                self.recenter_centroids(current_points_to_centroids)  # move centroids to mean of associated point 
                
        return self.cluster_centers_
    
    
    def calculate_distance(self, d_features: np.ndarray, c_features: np.ndarray) -> int:
        """
            Calculates the Euclidean distance between point A and point B. 
            Recall that a Euclidean distance can be expanded such that D^2 = A^2 + B^2 + ... Z^2. 
        """
        
        euclidean = 0
        for i in range(len(d_features)):
            euclidean += (math.pow(c_features[i] - d_features[i], 2))
      
        return math.sqrt(euclidean)

    def recenter_centroids(self, input: np.array) -> None:
        """
            This function recenters the centroid to the average distance of all its datapoints.
            Returns nothing, but updates cluster centers 
        """   
        for i in range(self.n_clusters):
            points = np.array(input[i])
            self.cluster_centers_[i] = np.nanmean(points)

        return None