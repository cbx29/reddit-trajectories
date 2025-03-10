#!/usr/bin/env python3
import os
import itertools
import numpy as np
import pandas as pd
from dtaidistance import dtw
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

def cluster_cities_with_dtw_pyclustering(
    city_files_folder,
    output_file,
    metric_name,
    resample_unit='W',
    max_clusters=10,
    n_jobs=-1,
    selected_cities=None
):
    """
    Performs DTW clustering on a specified metric (column) in city time series data 
    stored in parquet files using `pyclustering`.

    Parameters:
    - city_files_folder (str): Path to the folder containing city parquet files.
    - output_file (str): Path to save the combined results as a parquet file.
    - metric_name (str): The column name of the metric to cluster on.
    - resample_unit (str): Resampling frequency for time series (default 'W' = weekly).
    - max_clusters (int): Maximum number of clusters to test for the optimal number of clusters.
    - n_jobs (int): Number of parallel jobs (for distance matrix calculation). 
                    Default is -1 (use all cores).

    Saves:
    - A combined parquet file with the clustering results for all cities.
    """

    def load_city_data(file_path, metric):
        """
        Loads a city's metrics parquet file, extracts datetime values from column headings, 
        and converts it to a resampled and interpolated time series for the chosen metric only.
        """
        df = pd.read_parquet(file_path)
        
        try:
            df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
        except Exception as e:
            raise ValueError(f"Error converting column headers to datetime in {file_path}: {e}")
        
        df = df.T
        df.index.name = "datetime"

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        print(df.index.min(), df.index.max())

        df = df.loc["2020-01-01":"2021-12-31"]

        if metric not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in file columns. Available metrics: {list(df.columns)}"
            )
        
        df = df[[metric]]
        df = df.resample(resample_unit).mean().interpolate()
        return df.values.flatten()

    def compute_dtw_row(i, time_series_list):
        """
        Computes DTW distances from time series i to all subsequent series j>i
        to avoid double computation (since the distance matrix is symmetric).
        """
        n = len(time_series_list)
        row = np.zeros(n)
        for j in range(i + 1, n):
            row[j] = dtw.distance(time_series_list[i], time_series_list[j])
        return i, row

    def calculate_dtw_distance_matrix_parallel(time_series_list, n_jobs=-1):
        """
        Computes the pairwise DTW distance matrix in parallel using joblib.
        Returns an NxN numpy array of DTW distances.
        """
        n = len(time_series_list)
        distance_matrix = np.zeros((n, n))
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw_row)(i, time_series_list) for i in range(n)
        )
        
        for i, row in results:
            distance_matrix[i, i+1:] = row[i+1:]
            distance_matrix[i+1:, i] = row[i+1:]
        
        return distance_matrix

    def find_optimal_clusters(distance_matrix):
        """
        Try clustering with k=2..max_clusters, compute silhouette scores,
        pick k with the highest silhouette.
        """
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            initial_medoids = list(range(k))
            kmedoids_instance = kmedoids(distance_matrix.tolist(), initial_medoids)
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()

            labels = np.zeros(len(distance_matrix), dtype=int)
            for cluster_id, cluster_indices in enumerate(clusters):
                for idx in cluster_indices:
                    labels[idx] = cluster_id

            silhouette_scores.append(
                silhouette_score(distance_matrix, labels, metric="precomputed")
            )

        optimal_k = np.argmax(silhouette_scores) + 2
        return optimal_k, silhouette_scores

    print("Gathering city files...")
   
    all_files = [
        os.path.join(city_files_folder, f)
        for f in os.listdir(city_files_folder)
        if f.endswith('.parquet')
    ]
    all_city_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

    if selected_cities is not None:
        if isinstance(selected_cities, str):
            selected_cities = [selected_cities]
        selected_cities = [s.lower() for s in selected_cities]

        filtered_files = []
        filtered_names = []
        for f, name in zip(all_files, all_city_names):
            base_name = name.replace("_metrics", "").lower()
            if base_name in selected_cities:
                filtered_files.append(f)
                filtered_names.append(name)
        if not filtered_files:
            raise ValueError("None of the selected cities were found in the provided folder.")
        else:
            print(f"Selected cities found: {[n.replace('_metrics','') for n in filtered_names]}")
        city_files = filtered_files
        city_names = filtered_names
    else:
        city_files = all_files
        city_names = all_city_names


    output_dir = os.path.dirname(output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading city data for metric='{metric_name}'...")
    city_data = [load_city_data(f, metric_name) for f in city_files]

    print("Calculating DTW distance matrix...")
    distance_matrix = calculate_dtw_distance_matrix_parallel(city_data, n_jobs=n_jobs)

    print("Finding the optimal number of clusters...")
    optimal_clusters, silhouette_scores = find_optimal_clusters(distance_matrix)
    print(f"Optimal number of clusters by silhouette = {optimal_clusters}")

    print(f"Performing K-Medoids with k={optimal_clusters}...")
    initial_medoids = list(range(optimal_clusters))
    kmedoids_instance = kmedoids(distance_matrix.tolist(), initial_medoids)
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()

    cluster_labels = np.zeros(len(city_names), dtype=int)
    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            cluster_labels[idx] = cluster_id

    results_df = pd.DataFrame({
        "City": [name.replace("_metrics", "") for name in city_names],
        "Cluster": cluster_labels
    })

    print(f"Saving clustering results to {output_file}...")
    results_df.to_parquet(output_file, index=False)
    print("Done.")

    return optimal_clusters, silhouette_scores


if __name__ == "__main__":

    cluster_cities_with_dtw_pyclustering(
        city_files_folder="../metrics",
        output_file="../metric_clusters/comments_clusters.parquet",
        metric_name="Raw Number of Comments",
        resample_unit="W",
        max_clusters=10,
        n_jobs=-1
    )
