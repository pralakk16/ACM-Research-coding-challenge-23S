# Binary Star Systems in the Data

## Introduction

In this project, we are trying to find the Binary Star Systems in the star predication dataset using two clustering algorithms, K-Means and Agglomerative Clustering. Our goal is to compare the results of these two algorithms and see which one provides a better and more accurate picture of the Binary Star Systems.

## Import important packages and load data

We start by importing the important packages and load the dataset

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the star dataset
df = pd.read_csv('/kaggle/input/star-dataset/6 class csv.csv')

```

## Let's visualize the current data in a 3D Scatter Plot

```
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the temperature, relative luminosity, and relative radius of the stars
ax.scatter(df['Temperature (K)'], df['Luminosity(L/Lo)'], df['Radius(R/Ro)'])

# Label the axes
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Relative Luminosity (L/Lo)')
ax.set_zlabel('Relative Radius (R/Ro)')

# Show the plot
plt.show()
```

### We get the following visualization

![title](Images/initial_data.png)

## K-Means Clustering

We will now implement the The code for K-Means Clustering is as follows:

```
from sklearn.cluster import KMeans

# Extract the temperature, relative luminosity, and relative radius features
X = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)']]

# Train the KMeans algorithm
kmeans = KMeans(n_clusters=2).fit(X)

# Predict the cluster labels for each star
labels = kmeans.predict(X)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the stars in binary star systems in different colors using K-Means Algorithm
for label in np.unique(labels):
    ax.scatter(df[labels == label]['Temperature (K)'], df[labels == label]['Luminosity(L/Lo)'], df[labels == label]['Radius(R/Ro)'], label=f'Cluster {label}')

# Label the axes
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Relative Luminosity (L/Lo)')
ax.set_zlabel('Relative Radius (R/Ro)')

# Show the legend
plt.legend()

# Show the plot
plt.show()
```

### Here's what K-Means has given:

![title](Images/k_means.png)

### But we can do better than that!


## Agglomerative Clustering

Next, we implement Agglomerative Clustering on the same data. The code for Agglomerative Clustering is as follows:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load the star dataset
df = pd.read_csv('/kaggle/input/star-dataset/6 class csv.csv')

# Standardize the features
features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=2)
agg_cluster.fit(X_scaled)
labels = agg_cluster.labels_

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the stars in binary star systems in different colors using Hierarchial Agglomerative Clustering
for label in np.unique(labels):
    ax.scatter(df[labels == label]['Temperature (K)'], df[labels == label]['Luminosity(L/Lo)'], df[labels == label]['Radius(R/Ro)'], label=f'Cluster {label}')

# Label the axes
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Relative Luminosity (L/Lo)')
ax.set_zlabel('Relative Radius (R/Ro)')

# Show the legend
plt.legend()

# Show the plot
plt.show()

```

### The visualization of the Agglomerative Clustering results is as follows:

![title](Images/agglomerative.png)

### Much better!

## Comparison

From the visualizations above, we can see that the Agglomerative Clustering algorithm provides a better and more accurate picture of the Binary Star Systems as compared to K-Means Clustering. The Agglomerative Clustering algorithm is able to accurately distinguish between the two groups of Binary Star Systems, whereas K-Means Clustering fails to do so.

