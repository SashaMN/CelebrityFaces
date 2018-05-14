# CelebrityFaces
Finds similar celebrities given the photo.

You can test the service here: http://95.213.170.235:50002

# Example
Our algorithm:
![alt text](https://github.com/SashaMN/CelebrityFaces/raw/master/example.png)

Naive approach:
![alt text](https://github.com/SashaMN/CelebrityFaces/raw/master/naive.png)

# Algorithm
To build index:
1. Extract embeddings.
2. Cluster them with K-means algo into K=450 centroids.
3. Build inverse index: learn mapping from centroids to list of id of images.
4. Subtract centroid's center from corresponding images.
4. Build compressed descriptors: compute the sign of scalar products on 512 random directions and add small bias. This step reduces the size of descriptor from 128 to 8 64-bit numbers.

To process new image:
1. Load all the data in the memory from prev. algo.
2. Find top (K=10 out of 450) nearest centroids.
3. Compute Hamming distance in the compressed descriptors space between the query and all the images in the nearest centroids for each of 8 64-bit numbers.
4. Take top N=10 images from each table and sort them using full descriptions by euclidean distance.
5. Output top 5 images.

This algo implemented in C++.
