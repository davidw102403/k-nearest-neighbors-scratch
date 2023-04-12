import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class K_Nearest_Neighbors:

    def __init__(self, k=3):
        self.k = k

    def fit(self, points):
        self.points = points

    def euclidean_distance(self, p, q):
        return np.sqrt(np.sum(np.array(p) - np.array(q)) ** 2)

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = self.euclidean_distance(point, new_point)
                distances.append([distance, category])
        print(distances)

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
    
points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'orange': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [5,5]

clf = K_Nearest_Neighbors(k=3)
clf.fit(points)
print(clf.predict(new_point))