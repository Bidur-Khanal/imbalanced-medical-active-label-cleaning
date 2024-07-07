from __future__ import print_function, division
import numpy as np
from sklearn.metrics import pairwise_distances

# Adopted from https://github.com/svdesai/coreset-al/blob/master/coreset.py

class Coreset_Greedy:
    def __init__(self, all_pts):
      
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []


    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.all_pts[centers] # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
    
    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

       
        new_batch = []
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            
            assert ind not in already_selected
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)
        
        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return new_batch, max_distance