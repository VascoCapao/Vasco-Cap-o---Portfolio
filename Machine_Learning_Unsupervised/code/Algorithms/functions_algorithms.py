# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def visualize_dimensionality_reduction(transformation, targets):
    '''
    Create a scatter plot of a dimensionality reduction technique output.
    
    Arguments:
        - transformation (ndarray): A 2D numpy array representing the reduced dimensions of the data.
        - targets (array-like): An array-like object containing the target class labels for each sample. 
          The length of the array should be equal to the number of samples in the transformation array.
    
    Returns:
        None, but a scatter plot is displayed with points colored by their target classes.
    '''
    # create a scatter plot of the t-SNE output
    plt.scatter(transformation[:, 0], transformation[:, 1], 
              c=np.array(targets).astype(int), cmap=plt.cm.tab10)

    labels = np.unique(targets)

    # create a legend with the class labels and colors
    handles = [plt.scatter([],[], c=plt.cm.tab10(i), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Classes')

    plt.show()



def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    
    Arguments: 
        - model(HierarchicalClustering Model): hierarchical clustering model.
        - **kwargs
    
    Returns:
        None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def compute_silhouette_scores(data, range_n_clusters):
    """
    Computes the silhouette scores for a range of cluster numbers using the KMeans algorithm.

    Parameters:
        - data (array-like or sparse matrix): The input data.
        - range_n_clusters (iterable): The range of cluster numbers to consider.

    Returns:
        - list: A list of silhouette scores for each cluster number in the range.

    Notes:
        The silhouette score measures the compactness and separation of clusters. Higher scores indicate better-defined clusters.
    """
    # Create an empty list to store the silhouette scores
    silhouette_scores = []
    
    for n_clusters in range_n_clusters:
        # Initialize the KMeans model and fit it
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        # Predict the cluster for each data point
        cluster_labels = kmeans.labels_
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)

        print("For n_clusters =", n_clusters, "The average silhouette_score is:", silhouette_avg)
        
        # Append the score to the list
        silhouette_scores.append(silhouette_avg)
    
    return silhouette_scores


def visualize_data_points_grid(data, scaled_data, som_model, color_variable, color_dict):
  '''
  Plots scatter data points on top of a grid that represents the self-organizing map. 

  Each data point can be color coded with a "target" variable and we are plotting the distance map in the background.

  Arguments:
    - som_model(minisom.som): Trained self-organizing map.
    - color_variable(str): Name of the column to use in the plot.

  Returns:
    - None, but a plot is shown.
  '''

  # Subset target variable to color data points
  target = data[color_variable]

  fig, ax = plt.subplots()

  # Get weights for SOM winners
  w_x, w_y = zip(*[som_model.winner(d) for d in scaled_data])
  w_x = np.array(w_x)
  w_y = np.array(w_y)

  # Plot distance back on the background
  plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
  plt.colorbar()

  # Iterate through every data points - add some random perturbation just
  # to avoid getting scatters on top of each other.
  for c in np.unique(target):
      idx_target = target==c
      plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                  w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                  s=50, c=color_dict[c], label=c)

  ax.legend(bbox_to_anchor=(1.2, 1.05))
  plt.grid()
  plt.show()
