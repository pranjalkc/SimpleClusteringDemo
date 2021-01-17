"""
000731577, Pranjal Chauhan
This assignment is about machine learning techniques beyond supervised classification. Part A explores unsupervised clustering.
"""
#Elbow is between 4 and 6, so I'm choosing 5
import numpy as np
from skimage import data
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

##part 1 - plot a graph with different inertias and find an elbow
ks = list(range(2,20,2))
inertias = []
for k in ks:
    cow = io.imread("cow.JPG")
    cow = cow.reshape(339*251,3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(cow)
    inertia = kmeans.inertia_
    print("k =",k,", inertia =",inertia)
    inertias.append(inertia)


plt.figure()
plt.title("Plot of Inertia vs. K")
plt.xlabel("k")
plt.ylabel("inertia")
plt.plot(ks, inertias)
plt.show()

##part 2 - find color clustersand reload image with only those colors
cow = io.imread("cow.JPG")
image2 = io.imread("image2.JPG")
plt.figure()
plt.title("Before")
plt.imshow(cow)
plt.axis('off')
plt.show()
cow = cow.reshape(339*251,3)
image2 = image2.reshape(201*283,3)
#found elbow at 5
kmeans = KMeans(n_clusters=5, random_state=0).fit(image2)

colorClusters = kmeans.cluster_centers_
predictions = kmeans.predict(cow) #returns one of 5 numbers for each pixel
for i in range(5):
    cow[predictions==i] = colorClusters[i] #change each pixel with one of 5 clusters
cow = cow.reshape(339, 251, 3)
plt.figure()
plt.title("After clusters = 5")
plt.imshow(cow)
plt.axis('off')
plt.show()