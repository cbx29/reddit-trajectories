#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from time import sleep
from adjustText import adjust_text
import matplotlib.dates as mdates
from shapely.geometry import Point
import geopandas as gpd
from time import sleep


def plot_entries_per_week(json_file, output_dir, dataset_type):
    """
    Reads a JSON array file containing submission or comment data, counts the number per week,
    and saves a plot over time.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {json_file}: {e}")
            return

    if not data:
        print(f"No data to plot in file: {json_file}")
        return

    df = pd.DataFrame(data)

    if 'created_utc' not in df.columns:
        print(f"'created_utc' not found in data from file: {json_file}")
        return

    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df.set_index('created_utc', inplace=True)
    weekly_counts = df.resample('W').size()

    if weekly_counts.empty:
        print(f"No data to plot after resampling for file: {json_file}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(weekly_counts.index, weekly_counts.values, marker='o')
    plt.title(f'Number of {dataset_type.capitalize()} Per Week')
    plt.xlabel('Week')
    plt.ylabel(f'Number of {dataset_type.capitalize()}')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.splitext(os.path.basename(json_file))[0] + '_weekly.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()


def plot_entries_per_month(json_file, output_dir, dataset_type):
    """
    Reads a JSON array file containing submission or comment data, counts the number per week,
    and saves a plot over time.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {json_file}: {e}")
            return

    if not data:
        print(f"No data to plot in file: {json_file}")
        return

    df = pd.DataFrame(data)

    if 'created_utc' not in df.columns:
        print(f"'created_utc' not found in data from file: {json_file}")
        return

    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df.set_index('created_utc', inplace=True)
    monthly_counts = df.resample('MS').size()

    if monthly_counts.empty:
        print(f"No data to plot after resampling for file: {json_file}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts.index, monthly_counts.values, marker='o')
    plt.title(f'Number of {dataset_type.capitalize()} Per Month')
    plt.xlabel('Month')
    plt.ylabel(f'Number of {dataset_type.capitalize()}')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.splitext(os.path.basename(json_file))[0] + '_monthly.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()


def plot_comments_distribution(
    parquet_file_path,
    title='Comments Distribution',
    xlabel='Number of Comments',
    ylabel='Number of Posts',
    log_x=False,
    log_y=False,
    figsize=(10, 6),
    save_plot=False,
    plot_path='comments_distribution.png',
    display_plot=True
):
    """
    Plots the distribution of the number of comments on posts.

    Parameters:
    - parquet_file_path (str): Path to the Parquet file containing 'num_comments' field.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - log_x (bool): If True, set the x-axis to logarithmic scale.
    - log_y (bool): If True, set the y-axis to logarithmic scale.
    - figsize (tuple): Size of the plot figure.
    - save_plot (bool): If True, saves the plot to a file.
    - plot_path (str): File path to save the plot.

    Returns:
    - None
    """

    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"Error reading the Parquet file: {e}")
        return

    if 'num_comments' not in df.columns:
        print("The Parquet file does not contain a 'num_comments' column.")
        return

    df = df.dropna(subset=['num_comments'])
    df = df[df['num_comments'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]

    df['num_comments'] = df['num_comments'].astype(int)

    comments_counts = df['num_comments'].value_counts().sort_index()

    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")

    plt.scatter(comments_counts.index, comments_counts.values, alpha=0.6, edgecolor='b')

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    plt.tight_layout()

    if save_plot:
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_post_lifespan(
    posts_parquet_path,
    comments_parquet_path,
    units='hours',
    title='Post Lifespan Distribution',
    xlabel=None,
    ylabel='Number of Posts',
    return_data=False,
    log_scale=(False, False),
    verbose=False,
    plot_type='histogram',
    figsize=(10, 6),
    save_plot=False,
    plot_path='post_lifespan.png',
    display_plot=True
):
    """
    Calculates and plots the lifespan of posts based on comment activity,
    excluding posts with a lifespan of 0 (i.e., posts with only one comment).

    Parameters:
    - posts_parquet_path (str): Path to the posts Parquet file containing an 'id' field.
    - comments_parquet_path (str): Path to the comments Parquet file containing 'parent_id' and 'created_utc' fields.
    - units (str): Time units for lifespan. Options: 'seconds', 'minutes', 'hours', 'days'. Default is 'hours'.
    - title (str): Title of the plot. Default is 'Post Lifespan Distribution'.
    - xlabel (str): Label for the x-axis. If None, it will be set based on the 'units'.
    - ylabel (str): Label for the y-axis. Default is 'Number of Posts'.
    - figsize (tuple): Size of the plot figure. Default is (10, 6).
    - return_data (bool): If True, returns a DataFrame with post IDs and their lifespans. Default is False.
    - plot_type (str): Type of plot to generate. Options: 'histogram', 'density', 'cdf', 'box'. Default is 'histogram'.
    - log_scale (tuple): Tuple indicating whether to apply logarithmic scaling to the x and y axes.
                         Example: (True, False) applies log scale to x-axis only.
    - verbose (bool): If True, prints detailed logs for debugging purposes. Default is False.
    - save_plot (bool): If True, the plot will be saved to a file.
    - plot_path (str): File path to save the plot if save_plot is True.

    Returns:
    - If return_data=True, returns a pandas DataFrame with columns ['post_id', 'lifespan'].
    - Otherwise, returns None.
    """

    valid_units = ['seconds', 'minutes', 'hours', 'days']
    if units not in valid_units:
        raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")

    valid_plot_types = ['histogram', 'density', 'cdf', 'box']
    if plot_type not in valid_plot_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Choose from {valid_plot_types}.")

    if not (isinstance(log_scale, tuple) and len(log_scale) == 2 and 
            all(isinstance(x, bool) for x in log_scale)):
        raise ValueError("log_scale must be a tuple of two boolean values, e.g., (True, False).")

    if xlabel is None:
        xlabel = f'Lifespan ({units})'

    try:
        if verbose:
            print(f"Reading posts data from {posts_parquet_path}...")
        posts_df = pd.read_parquet(posts_parquet_path, columns=['id'])
    except Exception as e:
        print(f"Error reading the posts Parquet file: {e}")
        return

    if 'id' not in posts_df.columns:
        print("The posts Parquet file does not contain an 'id' column.")
        return

    if verbose:
        print(f"Number of posts loaded: {len(posts_df)}")

    try:
        if verbose:
            print(f"Reading comments data from {comments_parquet_path}...")
        comments_df = pd.read_parquet(comments_parquet_path, columns=['parent_id', 'created_utc'])
    except Exception as e:
        print(f"Error reading the comments Parquet file: {e}")
        return

    required_columns = ['parent_id', 'created_utc']
    for col in required_columns:
        if col not in comments_df.columns:
            print(f"The comments Parquet file does not contain a '{col}' column.")
            return

    if verbose:
        print(f"Number of comments loaded: {len(comments_df)}")

    comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)

    if verbose:
        print("Sample of cleaned 'parent_id':")
        print(comments_df['parent_id_clean'].head())

    if not np.issubdtype(comments_df['created_utc'].dtype, np.number):
        if verbose:
            print("'created_utc' is not numeric. Attempting to convert...")
        try:
            comments_df['created_utc'] = pd.to_numeric(comments_df['created_utc'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'created_utc' to numeric: {e}")
            return

    initial_comments = len(comments_df)
    comments_df = comments_df.dropna(subset=['created_utc'])
    if verbose:
        print(f"Dropped {initial_comments - len(comments_df)} comments due to invalid 'created_utc'.")

    if pd.api.types.is_datetime64_any_dtype(comments_df['created_utc']):
        if verbose:
            print("'created_utc' is datetime. Converting to Unix timestamp...")
        comments_df['created_utc'] = comments_df['created_utc'].astype(np.int64) // 10**9

    if verbose:
        print("Merging comments with posts...")
    merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner')

    if verbose:
        print(f"Number of comments after merging with posts: {len(merged_df)}")

    if verbose:
        print("Calculating min and max 'created_utc' for each post...")
    lifespan_df = merged_df.groupby('id')['created_utc'].agg(['min', 'max']).reset_index()
    lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

    if verbose:
        print(f"Number of posts with comments: {len(lifespan_df)}")
        print("Sample of lifespan data:")
        print(lifespan_df.head())

    lifespan_df['lifespan_seconds'] = lifespan_df['max'] - lifespan_df['min']

    invalid_lifespans = lifespan_df[lifespan_df['lifespan_seconds'] < 0]
    if not invalid_lifespans.empty:
        if verbose:
            print(f"Found {len(invalid_lifespans)} posts with negative lifespans. Excluding them.")
        lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

    if units == 'seconds':
        lifespan_df['lifespan'] = lifespan_df['lifespan_seconds']
    elif units == 'minutes':
        lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 60
    elif units == 'hours':
        lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 3600
    elif units == 'days':
        lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 86400

    if verbose:
        print(f"Converted lifespans to {units}.")
        print("Sample of converted lifespans:")
        print(lifespan_df['lifespan'].head())

    initial_count = len(lifespan_df)
    lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
    filtered_count = len(lifespan_df)
    excluded_posts = initial_count - filtered_count
    if verbose:
        print(f"Excluded {excluded_posts} posts with a lifespan of 0.")

    if lifespan_df.empty:
        print("No posts with lifespan greater than 0. Cannot plot histogram.")
        return

    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")

    if log_scale[0]:
        if lifespan_df['lifespan'].min() <= 0:
            lifespan_positive = lifespan_df['lifespan'] + 1e-6
            bins = np.logspace(np.log10(lifespan_positive.min()), np.log10(lifespan_positive.max()), 50)
        else:
            bins = np.logspace(np.log10(lifespan_df['lifespan'].min()), np.log10(lifespan_df['lifespan'].max()), 50)
    else:
        bins = 50

    if verbose:
        print(f"Plotting {plot_type} with {'logarithmic' if log_scale[0] else 'linear'} x-axis.")

    if plot_type == 'histogram':
        sns.histplot(
            lifespan_df['lifespan'],
            bins=bins,
            kde=False,
            color='skyblue',
            edgecolor='black'
        )
    elif plot_type == 'density':
        sns.kdeplot(
            lifespan_df['lifespan'],
            shade=True,
            color='skyblue'
        )
    elif plot_type == 'cdf':
        sns.ecdfplot(
            lifespan_df['lifespan'],
            color='skyblue'
        )
    elif plot_type == 'box':
        sns.boxplot(
            x=lifespan_df['lifespan'],
            color='skyblue'
        )
    else:
        raise ValueError(f"Unsupported plot_type '{plot_type}'.")

    if log_scale[0]:
        plt.xscale('log')
    if log_scale[1]:
        plt.yscale('log')

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.tight_layout()

    if save_plot:
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        if verbose:
            print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    if return_data:
        return lifespan_df[['post_id', 'lifespan']]


def load_city_data(file_path, metric, resample_unit, from_date, to_date):
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

    print("Data date range:", df.index.min(), "to", df.index.max())
    
    df = df.loc[from_date:to_date]

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in file columns. Available metrics: {list(df.columns)}"
        )
    
    df = df[[metric]]
    df = df.resample(resample_unit).mean().interpolate()
    return df.squeeze()


def plot_post_count(
    metric_file,
    city='',
    resample_unit='W',
    metric='Raw Number of Posts',
    save_plot=False,
    plot_path='post_count_timeseries.png',
    display_plot=True
):
    """
    Plots the time series for the number of posts over time by combining pre-COVID
    and main (COVID) datasets. This version loads the data using a precomputed metrics
    parquet file via the load_city_data function.
    
    Parameters
    ----------
    metric_file : str
        Path to the main metrics parquet file containing the post count metric.
    precovid_metric_file : str
        Path to the precovid metrics parquet file containing the post count metric.
    city : str, optional
        Name of the city for plot titling.
    resample_unit : str, optional
        Pandas resample frequency (default is 'W' for weekly).
    metric : str, optional
        The name of the metric column to use (default is 'post_count').
    save_plot : bool, optional
        Whether to save the plot to disk.
    plot_path : str, optional
        File path where the plot should be saved if save_plot is True.
    display_plot : bool, optional
        Whether to display the plot interactively.
    
    Returns
    -------
    None
    """
    print("Loading precovid posts count data...")
    precovid_counts = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")

    print("Loading main posts count data...")
    main_counts = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining datasets...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', label='Number of Posts', color='blue')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel('Number of Posts')
    plt.title(f'{city} Number of Posts Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    print("Done.")


def plot_comment_count(
    metric_file,
    city='',
    resample_unit='W',
    metric='comment_count',
    save_plot=False,
    plot_path='comment_count_timeseries.png',
    display_plot=True
):
    """
    Plots the time series for the number of comments over time by combining pre-COVID
    and main (COVID) datasets. This version loads the data using a precomputed metrics
    parquet file via the load_city_data function.
    
    Parameters
    ----------
    metric_file : str
        Path to the main metrics parquet file containing the comment count metric.
    precovid_metric_file : str
        Path to the pre-COVID metrics parquet file containing the comment count metric.
    city : str, optional
        Name of the city (if applicable) to be displayed in the plot title.
    resample_unit : str, optional
        Pandas resample frequency (default is 'W' for weekly).
    metric : str, optional
        The name of the metric column to use (default is 'comment_count').
    save_plot : bool, optional
        Whether to save the plot to disk.
    plot_path : str, optional
        File path where the plot should be saved if save_plot is True.
    display_plot : bool, optional
        Whether to display the plot interactively.
    
    Returns
    -------
    None
    """
    print("Loading pre-COVID comment count data...")
    precovid_counts = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")

    print("Loading main comment count data...")
    main_counts = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining datasets...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', label='Number of Comments', color='blue')

    # Mark the COVID start date
    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel('Number of Comments')
    plt.title(f'{city} Number of Comments Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    print("Done.")


def plot_comment_percentage_time_window(
    metric_file,
    city='',
    resample_unit='W',
    metric='comment_percentage_7day',
    save_plot=False,
    plot_path='comment_percentage_7day.png',
    display_plot=True
):
    """
    Plots the precomputed percentage of posts that received a comment within 7 days.
    It loads the main and precovid metrics from parquet files using load_city_data,
    concatenates them, and plots the combined time series with a fixed y-axis (0–100).
    """
    print("Loading precovid metrics data...")
    precovid_percentage = load_city_data(metric_file, metric, resample_unit, "2019-01-01", "2019-12-31")

    print("Loading main metrics data...")
    main_percentage = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining datasets...")
    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_percentage.plot(marker='o', linestyle='-', 
                               label='Percentage of Posts with Comments', color='blue')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', 
                label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel('Percentage of Posts with Comments Within 7 Days (%)')
    plt.title(f'{city} Percentage of Posts Receiving Comments Within 7 Days')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    print("Done.")


def plot_comment_percentage_time_window_normalised(
    metric_file,
    city='',
    resample_unit='W',
    metric='comment_percentage_7day',
    save_plot=False,
    plot_path='comment_percentage_7day.png',
    display_plot=True
):
    """
    Plots the precomputed normalized comment percentages.
    It loads the main and precovid metrics from parquet files using load_city_data,
    concatenates them, and plots the combined time series with y-axis settings that
    accommodate normalized (z-score) values.
    """
    print("Loading precovid metrics data...")
    precovid_percentage = load_city_data(metric_file, metric, resample_unit, "2019-01-01", "2019-12-31")

    print("Loading main metrics data...")
    main_percentage = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining datasets...")
    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_percentage.plot(marker='o', linestyle='-', 
                               label='Normalized Comment Percentage', color='blue')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', 
                label='COVID Start Date (Jan 2020)')

    plt.axhline(0, color='grey', linestyle=':', label='Baseline Mean (0)')

    plt.xlabel('Time')
    plt.ylabel('Normalized Comment Percentage (z-score)')
    plt.title(f'{city} Normalized Comment Percentage Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    print("Done.")


def plot_response_times_with_cutoff(
    metric_file,
    city='',
    resample_unit='W',
    metric='',
    time_diff_unit='minutes',
    save_plot=False,
    plot_path='average_response_times.png',
    display_plot=True
):
    """
    Plots the average response times of city subreddit posts, considering only responses
    that occur within 24 hours of a post, by combining pre-COVID and main (COVID) datasets.
    Data is loaded from precomputed metrics parquet files via load_city_data.
    
    The parquet files are assumed to have date strings (formatted as "%d/%m/%Y") as column headers
    and contain a column named 'average_response_time' (with values in seconds).
    
    Parameters:
    - metric_file (str): Path to the main metrics parquet file containing the average response time metric.
    - precovid_metric_file (str): Path to the pre-COVID metrics parquet file containing the average response time metric.
    - city (str): City name for the plot title.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - time_diff_unit (str): Unit for response time ('seconds', 'minutes', 'hours', or 'days').
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot if save_plot is True.
    - display_plot (bool): Whether to display the plot interactively.
    
    Returns:
    - None
    """
    print("Loading pre-COVID average response time data...")
    precovid_response = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")

    print("Loading main average response time data...")
    main_response = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining datasets...")
    combined_response = pd.concat([precovid_response, main_response])
    
    if time_diff_unit == 'seconds':
        response_converted = combined_response
        ylabel = 'Average Response Time (Seconds)'
    elif time_diff_unit == 'minutes':
        response_converted = combined_response / 60
        ylabel = 'Average Response Time (Minutes)'
    elif time_diff_unit == 'hours':
        response_converted = combined_response / 3600
        ylabel = 'Average Response Time (Hours)'
    elif time_diff_unit == 'days':
        response_converted = combined_response / 86400
        ylabel = 'Average Response Time (Days)'
    else:
        raise ValueError("Unsupported time_diff_unit. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

    print("Plotting the average response times with cutoff...")
    plt.figure(figsize=(12, 6))
    response_converted.plot(marker='o', linestyle='-', label='Average Response Time', color='blue')
    
    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')
    
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(f'{city} Average Response Times (Responses within 24 Hours) Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    
    if display_plot:
        plt.show()
    else:
        plt.close()
    
    print("Done.")


def plot_post_lifespan_timeseries(
    metric_file,
    resample_unit='weekly',
    metric='',
    units='hours',
    statistic='mean',
    title='Post Lifespan Over Time',
    xlabel=None,
    ylabel=None,
    figsize=(14, 7),
    return_data=False,
    log_scale=False,
    verbose=False,
    save_plot=False,
    plot_path='post_lifespan_timeseries.png',
    display_plot=True
):
    """
    Plots the post lifespan time series (aggregated by a specified interval) using precomputed metrics
    loaded from parquet files via load_city_data. Both pre‑COVID and main (COVID) data are loaded and combined.
    A red dotted vertical line is added at January 1, 2020.

    Parameters:
    -----------
    metric_file : str
        Path to the main metrics parquet file containing the post lifespan metric.
    precovid_metric_file : str
        Path to the pre‑COVID metrics parquet file containing the post lifespan metric.
    resample_unit : str, optional
        Aggregation interval; one of 'daily', 'weekly', 'monthly', 'yearly' (or a valid pandas resampling string).
        Default is 'monthly'.
    units : str, optional
        Time units for lifespan; one of 'seconds', 'minutes', 'hours', 'days'. Default is 'hours'.
        (Note: the precomputed metric is assumed to be in seconds for 'mean' or 'median'.)
    statistic : str, optional
        Statistic to plot; one of 'mean', 'median', 'count'. Default is 'mean'.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis. If None, it is set based on resample_unit.
    ylabel : str, optional
        Label for the y-axis. If None, it is set based on statistic and units.
    figsize : tuple, optional
        Figure size.
    return_data : bool, optional
        If True, returns the combined aggregated data as a DataFrame.
    log_scale : bool, optional
        If True, uses logarithmic scaling on the y-axis.
    verbose : bool, optional
        If True, prints detailed logs.
    save_plot : bool, optional
        If True, saves the plot to file.
    plot_path : str, optional
        File path to save the plot.
    display_plot : bool, optional
        If True, displays the plot interactively.

    Returns:
    --------
    pd.DataFrame or None
        The aggregated data (if return_data is True); otherwise, None.
    """

    resample_mapping = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'yearly': 'Y'
    }
    ru_lower = resample_unit.lower()
    if ru_lower in resample_mapping:
        resample_rule = resample_mapping[ru_lower]
    else:
        resample_rule = resample_unit

    if xlabel is None:
        xlabel = f"Time ({resample_unit.capitalize()})"
    if ylabel is None:
        if statistic in ['mean', 'median']:
            ylabel = f"Post Lifespan ({units.capitalize()})"
        elif statistic == 'count':
            ylabel = "Number of Posts"

    if verbose:
        print("Loading pre‑COVID post lifespan data...")
    precovid_series = load_city_data(metric_file, metric, resample_rule, "2019-10-01", "2019-12-31")

    if verbose:
        print("Loading main post lifespan data...")
    main_series = load_city_data(metric_file, metric, resample_rule, "2020-01-01", "2021-12-31")

    if verbose:
        print("Combining pre‑COVID and main data...")
    combined_series = pd.concat([precovid_series, main_series])
    
    if statistic in ['mean', 'median']:
        if units == 'seconds':
            conversion_factor = 1
        elif units == 'minutes':
            conversion_factor = 1/60
        elif units == 'hours':
            conversion_factor = 1/3600
        elif units == 'days':
            conversion_factor = 1/86400
        else:
            raise ValueError("Unsupported units. Choose from 'seconds', 'minutes', 'hours', or 'days'.")
        combined_series = combined_series * conversion_factor

    if verbose:
        print("Plotting aggregated post lifespan time series...")
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")

    if statistic in ['mean', 'median']:
        plt.plot(combined_series.index, combined_series.values, marker='o', linestyle='-', color='blue',
                 label=f'Post Lifespan ({statistic.capitalize()})')
    elif statistic == 'count':
        plt.bar(combined_series.index, combined_series.values, color='blue', label='Number of Posts')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if log_scale:
        plt.yscale('log')
    if resample_rule in ['M', 'Y']:
        plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        if verbose:
            print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()

    if verbose:
        print("Plotting complete.")

    if return_data:
        aggregated_df = combined_series.reset_index()
        aggregated_df.columns = ['Time', f'Post_Lifespan_{statistic}_{units}']
        return aggregated_df


def plot_post_sentiment(
    metric_file,
    city='',
    resample_unit='W',
    metric='average_sentiment',
    save_plot=False,
    plot_path='post_sentiment.png',
    display_plot=True
):
    """
    Plots the average sentiment of Reddit posts over time using precomputed metrics loaded via load_city_data().
    Combines pre‑COVID and main (COVID) datasets, and marks the COVID start date with a red dotted vertical line.
    
    Parameters:
    - metric_file (str): Path to the main metrics parquet file containing the sentiment metric.
    - precovid_metric_file (str): Path to the pre‑COVID metrics parquet file containing the sentiment metric.
    - city (str, optional): City name for the plot title.
    - resample_unit (str, optional): Pandas resample frequency (e.g., 'W' for weekly, 'M' for monthly).
    - metric (str, optional): Name of the sentiment metric column to load (default is 'average_sentiment').
    - save_plot (bool, optional): Whether to save the plot to a file.
    - plot_path (str, optional): File path where the plot should be saved.
    - display_plot (bool, optional): Whether to display the plot interactively.
    
    Returns:
    - None
    """
    print("Loading pre‑COVID post sentiment data...")
    precovid_sentiment = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")

    print("Loading main post sentiment data...")
    main_sentiment = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining pre‑COVID and main sentiment data...")
    combined_sentiment = pd.concat([precovid_sentiment, main_sentiment])

    print("Plotting the average sentiment over time...")
    plt.figure(figsize=(12, 6))
    combined_sentiment.plot(marker='o', linestyle='-', color='blue', label='Average Sentiment Score')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel('Average Sentiment Score')
    plt.title(f'{city} Average Sentiment of Posts Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_positive_sentiment_count(
    metric_file,
    city='',
    resample_unit='W',
    metric='positive_sentiment_count',
    threshold=0.05,
    save_plot=False,
    plot_path='positive_sentiment_count.png',
    display_plot=True
):
    """
    Plots the number of Reddit posts with positive sentiment (i.e., sentiment > threshold)
    over time using precomputed metrics loaded via load_city_data(). Both pre‑COVID and main (COVID)
    data are combined, and a red dotted vertical line marks the COVID start date (January 1, 2020).

    Parameters:
    - metric_file (str): Path to the main metrics parquet file containing the positive sentiment count metric.
    - precovid_metric_file (str): Path to the pre‑COVID metrics parquet file containing the positive sentiment count metric.
    - city (str, optional): City name or label for the plot title.
    - resample_unit (str, optional): Pandas resample frequency (e.g., 'W' for weekly, 'M' for monthly).
    - metric (str, optional): Name of the metric column to load (default is 'positive_sentiment_count').
    - threshold (float, optional): The sentiment threshold used (for reference in axis labels). Default is 0.05.
    - save_plot (bool, optional): Whether to save the plot to a file.
    - plot_path (str, optional): File path to save the plot.
    - display_plot (bool, optional): Whether to display the plot interactively.

    Returns:
    - None
    """
    print("Loading pre‑COVID positive sentiment count data...")
    # Load pre‑COVID data via load_city_data() and slice it to a fixed period (e.g., October 1, 2019 to December 31, 2019).
    precovid_counts = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")

    print("Loading main positive sentiment count data...")
    main_counts = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining pre‑COVID and main data...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    print(f"Plotting the number of posts with sentiment > {threshold} over time...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', color='blue', 
                         label=f'Posts with Sentiment > {threshold}')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', 
                label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel(f'Number of Posts (Sentiment > {threshold})')
    plt.title(f'{city} Positive Sentiment Post Count Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_negative_sentiment_count(
    metric_file,
    city='',
    resample_unit='W',
    metric='negative_sentiment_count',
    threshold=-0.05,
    save_plot=False,
    plot_path='negative_sentiment_count.png',
    display_plot=True
):
    """
    Plots the number of Reddit posts with negative sentiment (i.e. sentiment below a given threshold)
    over time using precomputed metrics loaded via load_city_data(). Pre‑COVID and main (COVID) data are
    combined, and a red dotted vertical line marks the COVID start date (January 1, 2020).

    Parameters:
    - metric_file (str): Path to the main metrics parquet file containing the negative sentiment count metric.
    - precovid_metric_file (str): Path to the pre‑COVID metrics parquet file containing the negative sentiment count metric.
    - city (str, optional): City name or label for the plot title.
    - resample_unit (str, optional): Pandas resample frequency (e.g., 'W' for weekly, 'M' for monthly).
    - metric (str, optional): Name of the metric column to load (default is 'negative_sentiment_count').
    - threshold (float, optional): Sentiment threshold used for labeling (not used in computation here). Default is -0.05.
    - save_plot (bool, optional): Whether to save the plot to a file.
    - plot_path (str, optional): File path to save the plot.
    - display_plot (bool, optional): Whether to display the plot interactively.

    Returns:
    - None
    """
    print("Loading pre‑COVID negative sentiment count data...")
    precovid_counts = load_city_data(metric_file, metric, resample_unit, "2019-10-01", "2019-12-31")
    
    print("Loading main negative sentiment count data...")
    main_counts = load_city_data(metric_file, metric, resample_unit, "2020-01-01", "2021-12-31")

    print("Combining pre‑COVID and main data...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    print(f"Plotting the number of posts with sentiment < {threshold} over time...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', color='blue', 
                         label=f'Posts with Sentiment < {threshold}')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', 
                label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel(f'Number of Posts (Sentiment < {threshold})')
    plt.title(f'{city} Negative Sentiment Post Count Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def get_global_y_limits(city_dict, metric_column):
    """
    Determines the global min and max values for a given metric across all cities.

    Parameters:
    - city_dict (dict): Dictionary where each key is a city identifier.
    - metric_column (str): The name of the metric column to extract min/max values.

    Returns:
    - (float, float): Global min and max values for the metric.
    """
    global_min = float('inf')
    global_max = float('-inf')

    for city_key in city_dict.keys():
        city_lower = city_key.lower().replace(" ", "")
        metric_parquet_path = f"../metrics/{city_lower}_metrics.parquet"
        
        if os.path.exists(metric_parquet_path):
            df = pd.read_parquet(metric_parquet_path)
            if metric_column in df.columns:
                global_min = min(global_min, df[metric_column].min())
                global_max = max(global_max, df[metric_column].max())

    return global_min, global_max


def plot_city_comparison(
    metric_file_city1,
    metric_file_city2,
    city1,
    city2,
    metric='comment_count',
    resample_unit='W',
    color1='blue',
    color2='green',
    covid_cutoff='2020-01-01',
    save_plot=False,
    plot_path='city_comparison_timeseries.png',
    display_plot=True
):
    """
    Overlays the same metric for two cities on one time‐series plot,
    with an optional COVID‐start vertical line.

    Parameters
    ----------
    metric_file_city1 : str
        Parquet file path for city1’s metrics.
    metric_file_city2 : str
        Parquet file path for city2’s metrics.
    city1 : str
        Name of the first city (used in legend/title).
    city2 : str
        Name of the second city.
    metric : str, optional
        Column name of the metric in both files.
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).
    color1, color2 : str, optional
        Plot colors for city1 & city2.
    covid_cutoff : str or pd.Timestamp, optional
        Date for vertical “COVID start” line.
    save_plot : bool, optional
        If True, saves the figure to `plot_path`.
    plot_path : str, optional
        File path to save the plot.
    display_plot : bool, optional
        If True, calls `plt.show()`, else closes it.

    Returns
    -------
    None
    """
    print(f"Loading {city1} data…")
    ts1 = load_city_data(metric_file_city1, metric, resample_unit, "2019-10-01", "2021-12-31")
    print(f"Loading {city2} data…")
    ts2 = load_city_data(metric_file_city2, metric, resample_unit, "2019-10-01", "2021-12-31")

    fig, ax = plt.subplots(figsize=(12, 6))
    ts1.plot(ax=ax, marker='o', linestyle='-', label=city1, color=color1)
    ts2.plot(ax=ax, marker='o', linestyle='-', label=city2, color=color2)

    covid_dt = pd.to_datetime(covid_cutoff)
    ax.axvline(covid_dt, color='red', linestyle=':', 
               label=f'COVID Start ({covid_dt.date()})')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} in {city1} vs {city2}')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_plot:
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


def generate_graphs_for_cities(city_dict):
    """
    Generates and saves graphs for each city contained in city_dict.

    Parameters:
    - city_dict (dict): Dictionary where each key is a city identifier (used in file paths and filenames)
                        and each value is the properly capitalized city name (for plot titles).

    Returns:
    - None: Graphs are saved as PNG files in the "graphs" folder.
    """
    print("Starting Graph Generation...")

    output_dir = "../graphs"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        metric_parquet_path = f"../metrics/{city_lower}_metrics.parquet"
        metric_normalised_parquet_path = f"../metrics_normalised/{city_lower}_metrics.parquet"
        liwc_metric_parquet_path = f"../liwc_metrics/{city_lower}_liwc_metrics.parquet"
        precovid_submissions_path = f"../precovid_data_parquet/{city_lower}_submissions.parquet"
        precovid_comments_path = f"../precovid_data_parquet/{city_lower}_comments.parquet"

        # calculate_and_plot_post_count(
        #     submissions_path, 
        #     comments_path,
        #     precovid_submissions_path,
        #     precovid_comments_path, 
        #     city=city_name, 
        #     resample_unit='W', 
        #     save_plot=True, 
        #     plot_path=f'../{output_dir}/{city_key}_post_count_timeseries.png', 
        #     display_plot=False
        # )

        # calculate_and_plot_comment_count(
        #     submissions_path, 
        #     comments_path,
        #     precovid_submissions_path, 
        #     precovid_comments_path, 
        #     city=city_name, 
        #     resample_unit='W', 
        #     save_plot=True, 
        #     plot_path=f'../{output_dir}/{city_key}_comment_count_timeseries.png', 
        #     display_plot=False
        # )

        # --- Plot comment percentage ---
        # calculate_and_plot_comment_percentage_time_window(
        #     submissions_path,
        #     comments_path,
        #     precovid_submissions_path,
        #     precovid_comments_path,
        #     city=city_name,
        #     resample_unit='W',
        #     save_plot=True,
        #     plot_path=f'../{output_dir}/{city_key}_comment_percentage_same_week.png',
        #     display_plot=False
        # )

        # --- Plot response times ---
        # calculate_and_plot_response_times_with_cutoff(
        #     submissions_path,
        #     comments_path,
        #     city=city_name,
        #     time_unit='W',
        #     time_diff_unit='hours',
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_average_response_times_cutoff.png',
        #     display_plot=False
        # )

        # # --- Plot response times ---

        # # --- Plot post lifespan timeseries ---
        # calculate_and_plot_post_lifespan_timeseries(
        #     submissions_path,
        #     comments_path,
        #     aggregation='weekly',
        #     units='hours',
        #     statistic='mean',
        #     title=f'{city_name} Post Lifespan Timeseries',
        #     xlabel='Week',
        #     ylabel='Average Lifespan (Hours)',
        #     figsize=(16, 8),
        #     return_data=False,
        #     log_scale=False,
        #     verbose=False,
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_post_lifespan_timeseries.png',
        #     display_plot=False
        # )

        # # --- Plot post lifespan distribution ---
        # plot_post_lifespan(
        #     submissions_path,
        #     comments_path,
        #     units='hours',
        #     title=f'{city_name} Post Lifespan Distribution',
        #     xlabel='Log Lifespan (Hours)',
        #     ylabel='Number of Posts',
        #     return_data=False,
        #     log_scale=(True, False),
        #     verbose=False,
        #     plot_type='histogram',
        #     figsize=(10, 6),
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_post_lifespan.png',
        #     display_plot=False
        # )
        # plt.clf()

        # # --- Plot comments distribution ---
        # plot_comments_distribution(
        #     submissions_path,
        #     title=f'{city_name} Comments Distribution',
        #     xlabel='Log Number of Comments',
        #     ylabel='Log Number of Posts',
        #     log_x=True,
        #     log_y=True,
        #     figsize=(10, 6),
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_comments_distribution.png',
        #     display_plot=False
        # )
        # plt.clf()

        # # --- Plot post sentiment ---
        # calculate_and_plot_post_sentiment(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric='average_sentiment',
        #     save_plot=False,
        #     plot_path='post_sentiment.png',
        #     display_plot=True
        # )

        # calculate_and_plot_positive_sentiment_count(
        #     submissions_path,
        #     city=city_name,
        #     text_column='selftext',
        #     time_unit='W',      
        #     threshold=0.05,          
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_positive_sentiment_count.png',
        #     display_plot=False
        # )
        # plt.clf()

        # calculate_and_plot_negative_sentiment_count(
        #     submissions_path,
        #     city=city_name,
        #     text_column='selftext',
        #     time_unit='W',       
        #     threshold=-0.05,            
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_negative_sentiment_count.png',
        #     display_plot=False
        # )
        # plt.clf()

        # ------ WITHOUT RECALCULATING ------- #

        # plot_post_count(
        #     metric_parquet_path,
        #     city=city_name, 
        #     resample_unit='W', 
        #     metric="Raw Number of Posts",
        #     save_plot=True, 
        #     plot_path=f'{output_dir}/{city_key}_post_count_timeseries.png', 
        #     display_plot=False
        # )

        # plot_comment_count(
        #     metric_parquet_path,
        #     city=city_name, 
        #     resample_unit='W', 
        #     metric="Raw Number of Comments",
        #     save_plot=True, 
        #     plot_path=f'{output_dir}/{city_key}_comment_count_timeseries.png', 
        #     display_plot=False
        # )

        # plot_comment_percentage_time_window(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric="Percentage of Posts with Comments",
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_comment_percentage_same_week.png',
        #     display_plot=False
        # )

        # plot_response_times_with_cutoff(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric = 'Average Response Time with Cutoff (Minutes)',
        #     time_diff_unit='minutes',
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_average_response_times.png',
        #     display_plot=False
        # )

        # plot_post_lifespan_timeseries(
        #     metric_parquet_path,
        #     resample_unit='weekly',
        #     metric='Post Lifespan (Mean in hours)',
        #     units='hours',
        #     statistic='mean',
        #     title='Post Lifespan Over Time',
        #     xlabel=None,
        #     ylabel=None,
        #     figsize=(14, 7),
        #     return_data=False,
        #     log_scale=False,
        #     verbose=False,
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_post_lifespan_timeseries.png',
        #     display_plot=False
        # )

        # plot_post_sentiment(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric='Average Sentiment',
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_post_sentiment.png',
        #     display_plot=False
        # )

        # plot_positive_sentiment_count(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric='Positive Sentiment Count',
        #     threshold=0.05,
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_positive_sentiment_count.png',
        #     display_plot=False
        # )

        # plot_negative_sentiment_count(
        #     metric_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric='Negative Sentiment Count',
        #     threshold=-0.05,
        #     save_plot=True,
        #     plot_path=f'{output_dir}/{city_key}_negative_sentiment_count.png',
        #     display_plot=False
        # )

        # plot_comment_percentage_time_window_normalised(
        #     metric_normalised_parquet_path,
        #     city=city_name,
        #     resample_unit='W',
        #     metric="Percentage of Posts with Comments",
        #     save_plot=True,
        #     plot_path=f'../comment_percentage_graphs_2/{city_key}_comment_percentage.png',
        #     display_plot=False
        # )

        plot_post_count(
            metric_parquet_path,
            city=city_name, 
            resample_unit='W', 
            metric="Raw Number of Posts",
            save_plot=True, 
            plot_path=f'../report_graphs/post_count/{city_key}_post_count_timeseries.png', 
            display_plot=False
        )

        plot_comment_count(
            metric_parquet_path,
            city=city_name, 
            resample_unit='W', 
            metric="Raw Number of Comments",
            save_plot=True, 
            plot_path=f'../report_graphs/comment_count/{city_key}_comment_count_timeseries.png', 
            display_plot=False
        )

        plot_comment_percentage_time_window(
            metric_parquet_path,
            city=city_name,
            resample_unit='W',
            metric="Percentage of Posts with Comments",
            save_plot=True,
            plot_path=f'../report_graphs/comment_percentage/{city_key}_comment_percentage_same_week.png',
            display_plot=False
        )

        plot_response_times_with_cutoff(
            metric_parquet_path,
            city=city_name,
            resample_unit='W',
            metric = 'Average Response Time with Cutoff (Minutes)',
            time_diff_unit='minutes',
            save_plot=True,
            plot_path=f'../report_graphs/response/{city_key}_average_response_times.png',
            display_plot=False
        )

        plot_post_lifespan_timeseries(
            metric_parquet_path,
            resample_unit='weekly',
            metric='Post Lifespan (Mean in hours)',
            units='hours',
            statistic='mean',
            title='Post Lifespan Over Time',
            xlabel=None,
            ylabel=None,
            figsize=(14, 7),
            return_data=False,
            log_scale=False,
            verbose=False,
            save_plot=True,
            plot_path=f'../report_graphs/lifespan/{city_key}_post_lifespan_timeseries.png',
            display_plot=False
        )

        plot_post_sentiment(
            metric_parquet_path,
            city=city_name,
            resample_unit='W',
            metric='Average Sentiment',
            save_plot=True,
            plot_path=f'../report_graphs/average_sentiment//{city_key}_post_sentiment.png',
            display_plot=False
        )

        plot_positive_sentiment_count(
            metric_parquet_path,
            city=city_name,
            resample_unit='W',
            metric='Positive Sentiment Count',
            threshold=0.05,
            save_plot=True,
            plot_path=f'../report_graphs/positive_sentiment/{city_key}_positive_sentiment_count.png',
            display_plot=False
        )

        plot_negative_sentiment_count(
            metric_parquet_path,
            city=city_name,
            resample_unit='W',
            metric='Negative Sentiment Count',
            threshold=-0.05,
            save_plot=True,
            plot_path=f'../report_graphs/negative_sentiment/{city_key}_negative_sentiment_count.png',
            display_plot=False
        )

        print(f"Graphs for {city_name} saved in the '{output_dir}' folder.")


def plot_two_clusters_timeseries(parquet_file_path, save_path=None):
    df = pd.read_parquet(parquet_file_path)

    if '_index_level_0' in df.columns:
        df = df.set_index('_index_level_0')

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df['Cluster 0'], label='Cluster 0', color='blue')
    plt.plot(df.index, df['Cluster 1'], label='Cluster 1', color='orange')

    plt.title('Aggregated Percentage of Posts Receiving Comments based on Clusters')
    plt.xlabel('Time')
    plt.ylabel('Percentage of posts receiving comments')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_liwc_clusters_timeseries(parquet_file_path, title, yaxis, start_date, end_date=None, save_path=None):
    df = pd.read_parquet(parquet_file_path)

    if '_index_level_0' in df.columns:
        df = df.set_index('_index_level_0')

    df.index = pd.to_datetime(df.index, format="%d/%m/%Y")

    df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]

    plt.figure(figsize=(10, 6))
    
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    
    covid_date = pd.to_datetime("01/01/2020", format="%d/%m/%Y")
    plt.axvline(x=covid_date, color='red', linestyle=':', linewidth=1.5, label='COVID-19 Start')
    plt.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Zero Line')
    
    plt.title(title, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(yaxis, fontsize=25)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [label.replace("LIWC Percentage for ", "") for label in labels]
    plt.legend(handles, new_labels, fontsize=20)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_shape_vs_scale(parquet_file_path, save_path=None):
    """
    Reads a parquet file containing columns:
    'category', 'scale_difference', 'shape_difference', and 'classification',
    and then plots a scatter plot with:
      - x-axis: shape_difference
      - y-axis: scale_difference
    Each point is labeled with its category name.
    Additionally, vertical and horizontal dotted lines are drawn at 0.5 on both axes.
    The x- and y-axes are set to always start at 0.
    
    Parameters:
      parquet_file_path (str): Path to the parquet file.
      save_path (str, optional): If provided, the plot will be saved to this path.
    """
    df = pd.read_parquet(parquet_file_path)
    
    plt.figure(figsize=(10, 8))
    
    for _, row in df.iterrows():
        shape = row['shape_difference']
        scale = row['scale_difference']
        category = row['category']
        plt.scatter(shape, scale, color='blue', s=50)
        plt.text(shape, scale, f' {category}', fontsize=9, ha='left', va='center')
    
    plt.axvline(x=0.5, color='red', linestyle=':', linewidth=1.5)
    plt.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5)
    
    plt.xlabel("Shape Difference")
    plt.ylabel("Scale Difference")
    plt.title("Shape vs. Scale Differences by Category")
    plt.grid(True)
    
    ax = plt.gca()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_shape_vs_scale_knn(parquet_file_path, save_path=None, n_neighbors=3):
    """
    Reads a parquet file containing columns:
    'category', 'scale_difference', 'shape_difference', and 'classification'.
    
    It then:
      - Encodes the classification labels to numeric values.
      - Trains a KNN classifier (default n_neighbors=3) using shape_difference and scale_difference as features.
      - Computes predicted classifications on a grid over the feature space.
      - Plots the decision boundaries as a contour fill.
      - Overlays a scatter plot of the data, labeling each point with its category name.
      - Adjusts text positions to avoid overlaps.
      - Draws vertical and horizontal dotted lines at 0.5.
      - Sets both x- and y-axes to start at 0 and end at the largest observed values (or 0.5 if larger values aren't present).
      
    Parameters:
      parquet_file_path (str): Path to the parquet file.
      save_path (str, optional): If provided, the plot will be saved to this path.
      n_neighbors (int, optional): Number of neighbors to use for KNN.
    """

    df = pd.read_parquet(parquet_file_path)
    
    X = df[['shape_difference', 'scale_difference']].values
    y = df['classification'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y_encoded)
    
    x_max = 1.1 * max(df['shape_difference'].max(), 0.5)
    y_max = 1.1 * max(df['scale_difference'].max(), 0.5)

    xx, yy = np.meshgrid(np.linspace(0, x_max, 200),
                         np.linspace(0, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

    texts = []
    for _, row in df.iterrows():
        shape = row['shape_difference']
        scale = row['scale_difference']
        category = row['category']
        plt.scatter(shape, scale, color='blue', s=50)
        t = plt.text(shape, scale, f' {category}', fontsize=12, ha='left', va='center')
        texts.append(t)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=1), force_text=0.5)
    
    plt.axvline(x=0.5, color='red', linestyle=':', linewidth=1.5)
    plt.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5)
    
    plt.xlabel("Shape Difference", fontsize=16)
    plt.ylabel("Scale Difference", fontsize=16)
    plt.title("Shape vs. Scale Differences by Category with KNN Decision Boundaries", fontsize=18)
    plt.grid(True)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_word_shifts_from_parquet(filepath, category, top_n=50, save=False, filename='word_shifts.png'):
    """
    Reads a Parquet file (with 'words' and 'percentage_change' columns) 
    and produces a horizontal bar chart of the top word shifts, sorted in descending order
    by the absolute value of percentage_change.

    Parameters
    ----------
    filepath : str
        Path to the Parquet file.
    top_n : int
        Number of words to display (sorted by absolute change).
    save : bool
        If True, saves the figure to 'filename' instead of displaying it.
    filename : str
        The file path to save the figure when save=True.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_parquet(filepath)

    df_top = df.sort_values('percentage_change', key=lambda x: x.abs(), ascending=False).head(top_n)

    colors = df_top['percentage_change'].apply(lambda x: 'darkorange' if x > 0 else 'purple')

    fig, ax = plt.subplots(figsize=(7, 10))

    ax.barh(df_top['words'], df_top['percentage_change'], color=colors)

    ax.invert_yaxis()

    ax.axvline(x=0, color='black', linewidth=1)

    ax.set_xlabel('Score shift (Δp, %)', fontsize=20)
    ax.set_ylabel('', fontsize=20)
    ax.set_title(f'Top {top_n} {category} Word Shifts', fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Plot saved to {filename}")
    else:
        plt.show()


def plot_us_cities_map_cartopy(
    city_dict,
    title="Final 85 Selected U.S. Cities for Subreddit Study",
    annotate=False,
    save_path="../us_city_map.png",
    marker_size=30,
    marker_color="red",
    sleep_time=1
):
    """
    Plots a static U.S. map with cities from city_dict using Cartopy and saves it as an image.

    Parameters:
        city_dict (dict): Mapping from city keys to full city names (e.g., "newyorkcity": "New York City").
        title (str): Title of the map.
        annotate (bool): Whether to annotate city names.
        save_path (str): File path to save the output image (e.g., "map.png").
        marker_size (int): Size of the city markers.
        marker_color (str): Color of the city markers.
        sleep_time (int): Seconds to wait between geocoding requests.
    """
    geolocator = Nominatim(user_agent="city_mapper")
    cities = []

    print("Geocoding cities...")
    for key, name in city_dict.items():
        try:
            location = geolocator.geocode(f"{name}, USA")
            if location:
                cities.append({
                    "CityKey": key,
                    "CityName": name,
                    "Latitude": location.latitude,
                    "Longitude": location.longitude
                })
            else:
                print(f"Could not locate: {name}")
        except GeocoderTimedOut:
            print(f"Timed out: {name}")
        except Exception as e:
            print(f"Error with {name}: {e}")
        sleep(sleep_time)

    df = pd.DataFrame(cities)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=39))
    ax.set_extent([-130, -65, 23, 50], crs=ccrs.Geodetic())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4)
    ax.coastlines(resolution='50m', linewidth=0.5)

    ax.scatter(
        df["Longitude"],
        df["Latitude"],
        color=marker_color,
        s=marker_size,
        alpha=0.8,
        transform=ccrs.Geodetic(),
        label="Selected Cities"
    )

    if annotate:
        for _, row in df.iterrows():
            ax.text(
                row["Longitude"],
                row["Latitude"],
                row["CityName"],
                transform=ccrs.Geodetic(),
                fontsize=6,
                ha='left',
                va='center'
            )

    plt.title(title, fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Map saved to: {save_path}")
    plt.close()


def plot_us_cities_map_from_dict(
    city_dict,
    title="Selected U.S. Cities Used in the Study",
    annotate=False,
    save_path=None,
    marker_color="red",
    marker_size=30,
    sleep_time=1
):
    """
    Plots a U.S. map with cities from a city_dict marked as dots.

    Parameters:
        city_dict (dict): Dictionary mapping city keys (e.g., "newyorkcity") to full names (e.g., "New York City").
        title (str): Title of the plot.
        annotate (bool): Whether to annotate city names on the map.
        save_path (str): Optional file path to save the plot (e.g., "us_map.png").
        marker_color (str): Color of city markers.
        marker_size (int): Size of city markers.
        sleep_time (int): Delay between geocoding requests (to avoid rate limiting).
    """

    geolocator = Nominatim(user_agent="city_mapper")
    city_coords = []

    print("Geocoding cities...")
    for key, city in city_dict.items():
        try:
            location = geolocator.geocode(f"{city}, USA")
            if location:
                city_coords.append((key, city, location.latitude, location.longitude))
            else:
                print(f"Could not locate: {city}")
        except Exception as e:
            print(f"Error with city {city}: {e}")
        sleep(sleep_time)

    df = pd.DataFrame(city_coords, columns=["CityKey", "CityName", "Latitude", "Longitude"])
    df["geometry"] = df.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    usa = gpd.read_file("../ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    usa = usa[usa["ADMIN"] == "United States of America"]

    fig, ax = plt.subplots(figsize=(12, 8))
    usa.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf.plot(ax=ax, color=marker_color, markersize=marker_size)

    if annotate:
        for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf["CityName"]):
            ax.text(x, y, label, fontsize=6)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Map saved to {save_path}")

    plt.show()


def plot_response_time_comparison(
    metric_file_city1,
    metric_file_city2,
    city1,
    city2,
    resample_unit='W',
    metric='average_response_time',
    time_diff_unit='minutes',
    covid_cutoff='2020-01-01',
    save_plot=False,
    plot_path='comparison_response_times.png',
    display_plot=True
):
    """
    Overlays average response times for two cities on the same axis,
    including only responses within 24h, and marks the COVID start date.

    Parameters
    ----------
    metric_file_city1 : str
        Parquet file for city1’s metrics.
    metric_file_city2 : str
        Parquet file for city2’s metrics.
    city1 : str
        Name of the first city.
    city2 : str
        Name of the second city.
    resample_unit : str, optional
        Pandas resample frequency ('W', 'M', etc.).
    metric : str, optional
        Column name of the response time metric (in seconds).
    time_diff_unit : str, optional
        Unit to convert response time into:
        'seconds', 'minutes', 'hours', or 'days'.
    covid_cutoff : str or pd.Timestamp, optional
        Date to draw vertical line (default '2020-01-01').
    save_plot : bool, optional
        If True, saves the plot to `plot_path`.
    plot_path : str, optional
        File path to save the plot.
    display_plot : bool, optional
        If True, calls `plt.show()`, else closes figure.

    Returns
    -------
    None
    """
    def load_and_concat(fpath):
        pre = load_city_data(fpath, metric, resample_unit, "2019-10-01", "2019-12-31")
        main = load_city_data(fpath, metric, resample_unit, "2020-01-01", "2021-12-31")
        return pd.concat([pre, main])

    print(f"Loading response times for {city1}…")
    ts1 = load_and_concat(metric_file_city1)
    print(f"Loading response times for {city2}…")
    ts2 = load_and_concat(metric_file_city2)

    # Unit conversion
    if time_diff_unit == 'seconds':
        factor, ylabel = 1, 'Average Response Time (Seconds)'
    elif time_diff_unit == 'minutes':
        factor, ylabel = 1/60, 'Average Response Time (Minutes)'
    elif time_diff_unit == 'hours':
        factor, ylabel = 1/3600, 'Average Response Time (Hours)'
    elif time_diff_unit == 'days':
        factor, ylabel = 1/86400, 'Average Response Time (Days)'
    else:
        raise ValueError("Unsupported time_diff_unit: choose 'seconds','minutes','hours','days'")

    ts1 = ts1 * factor
    ts2 = ts2 * factor

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ts1.plot(ax=ax, marker='o', linestyle='-', label=city1, color='blue')
    ts2.plot(ax=ax, marker='o', linestyle='-', label=city2, color='green')

    covid_dt = pd.to_datetime(covid_cutoff)
    ax.axvline(covid_dt, color='red', linestyle=':', label=f'COVID Start ({covid_dt.date()})')

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} in {city1} vs {city2}')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_plot:
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


def plot_post_lifespan_three_way(
    metric_file_city1,
    metric_file_city2,
    metric_file_city3,
    city1,
    city2,
    city3,
    resample_unit='weekly',
    metric='post_lifespan_seconds',
    units='hours',
    statistic='mean',
    covid_cutoff='2020-01-01',
    figsize=(14, 7),
    log_scale=False,
    save_plot=False,
    plot_path='post_lifespan_three_way.png',
    display_plot=True,
    return_data=False,
    verbose=False
):
    """
    Overlays post lifespan stats for three cities on the same axis,
    combining pre-COVID and main (COVID) data, with a vertical COVID start marker.

    Parameters
    ----------
    metric_file_city1, metric_file_city2, metric_file_city3 : str
        Parquet files for each city’s metrics.
    city1, city2, city3 : str
        Names of the three cities.
    resample_unit : str, optional
        'daily', 'weekly', 'monthly', 'yearly' or any pandas resample rule.
    metric : str, optional
        Column name for the lifespan metric (in seconds).
    units : str, optional
        One of 'seconds', 'minutes', 'hours', 'days'.
    statistic : str, optional
        'mean', 'median', or 'count'.
    covid_cutoff : str or pd.Timestamp, optional
        Date for vertical line.
    figsize : tuple, optional
        Figure size.
    log_scale : bool, optional
        If True, y-axis is log-scaled.
    save_plot : bool, optional
        If True, saves figure to `plot_path`.
    plot_path : str, optional
        Path to save the plot.
    display_plot : bool, optional
        If True, calls plt.show().
    return_data : bool, optional
        If True, returns the combined DataFrame.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    pd.DataFrame or None
    """
    rm = {'daily':'D','weekly':'W','monthly':'M','yearly':'Y'}
    rule = rm.get(resample_unit.lower(), resample_unit)

    def load_city(fpath):
        if verbose: print(f"Loading pre-COVID data for {fpath}…")
        pre = load_city_data(fpath, metric, rule, "2019-10-01", "2019-12-31")
        if verbose: print(f"Loading main data for {fpath}…")
        main = load_city_data(fpath, metric, rule, "2020-01-01", "2021-12-31")
        return pd.concat([pre, main])

    ts1 = load_city(metric_file_city1)
    ts2 = load_city(metric_file_city2)
    ts3 = load_city(metric_file_city3)

    if statistic in ('mean','median'):
        conv = {'seconds':1, 'minutes':1/60, 'hours':1/3600, 'days':1/86400}
        if units not in conv:
            raise ValueError("Unsupported units: choose 'seconds','minutes','hours','days'")
        factor = conv[units]
        ts1, ts2, ts3 = ts1*factor, ts2*factor, ts3*factor
        ylabel = f"Post Lifespan ({units})"
    else:
        ylabel = "Number of Posts"

    fig, ax = plt.subplots(figsize=figsize)
    if statistic in ('mean','median'):
        ax.plot(ts1.index, ts1.values, marker='o', linestyle='-',  label=city1, color='blue')
        ax.plot(ts2.index, ts2.values, marker='o', linestyle='-', label=city2, color='green')
        ax.plot(ts3.index, ts3.values, marker='o', linestyle='-',  label=city3, color='orange')
    else:
        width = 0.8 * pd.to_timedelta(1, unit=rule)
        ax.bar(ts1.index - width/3, ts1.values, width=width/3, label=city1, alpha=0.7)
        ax.bar(ts2.index,            ts2.values, width=width/3, label=city2, alpha=0.7)
        ax.bar(ts3.index + width/3, ts3.values, width=width/3, label=city3, alpha=0.7)

    cv = pd.to_datetime(covid_cutoff)
    ax.axvline(cv, color='red', linestyle=':', label=f'COVID Start ({cv.date()})')

    ax.set_title(f"Post Lifespan ({statistic.capitalize()}) — {city1}, {city2} & {city3}")
    ax.set_xlabel(f"Time ({resample_unit.capitalize()})")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
        fig.savefig(plot_path)
        if verbose: print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    if return_data:
        df = pd.DataFrame({
            f"{city1}_{statistic}_{units}": ts1,
            f"{city2}_{statistic}_{units}": ts2,
            f"{city3}_{statistic}_{units}": ts3
        })
        df.index.name = 'Time'
        return df

    if verbose: print("Done.")


if __name__ == "__main__":

    city_dict = {
        "albuquerque": "Albuquerque",
        "anchorage": "Anchorage",
        "arlington": "Arlington",
        "atlanta": "Atlanta",
        "aurora": "Aurora",
        "austin": "Austin",
        "bakersfield": "Bakersfield",
        "baltimore": "Baltimore",
        "batonrouge": "Baton Rouge",
        "boise": "Boise",
        "boston": "Boston",
        "buffalo": "Buffalo",
        "charlotte": "Charlotte",
        "chicago": "Chicago",
        "cincinnati": "Cincinnati",
        "cleveland": "Cleveland",
        "coloradosprings": "Colorado Springs",
        "columbus": "Columbus",
        "corpuschristi": "Corpus Christi",
        "dallas": "Dallas",
        "denver": "Denver",
        "detroit": "Detroit",
        "elpaso": "El Paso",
        "fortwayne": "Fort Wayne",
        "fortworth": "Fort Worth",
        "fremont": "Fremont",
        "fresno": "Fresno",
        "glendale": "Glendale",
        "honolulu": "Honolulu",
        "houston": "Houston",
        "indianapolis": "Indianapolis",
        "irvine": "Irvine",
        "jacksonville": "Jacksonville",
        "jerseycity": "Jersey City",
        "kansascity": "Kansas City",
        "laredo": "Laredo",
        "lasvegas": "Las Vegas",
        "lexington": "Lexington",
        "lincoln": "Lincoln",
        "longbeach": "Long Beach",
        "losangeles": "Los Angeles",
        "louisville": "Louisville",
        "lubbock": "Lubbock",
        "madison": "Madison",
        "memphis": "Memphis",
        "miami": "Miami",
        "milwaukee": "Milwaukee",
        "minneapolis": "Minneapolis",
        "nashville": "Nashville",
        "newark": "Newark",
        "neworleans": "New Orleans",
        "norfolk": "Norfolk",
        "newyorkcity": "New York City",
        "oakland": "Oakland",
        "oklahomacity": "Oklahoma City",
        "omaha": "Omaha",
        "orlando": "Orlando",
        "philadelphia": "Philadelphia",
        "pittsburgh": "Pittsburgh",
        "plano": "Plano",
        "portland": "Portland",
        "reno": "Reno",
        "richmond": "Richmond",
        "riverside": "Riverside",
        "sacramento": "Sacramento",
        "saintpaul": "Saint Paul",
        "sanantonio": "San Antonio",
        "sandiego": "San Diego",
        "sanfrancisco": "San Francisco",
        "sanjose": "San Jose",
        "santaclarita": "Santa Clarita",
        "scottsdale": "Scottsdale",
        "seattle": "Seattle",
        "spokane": "Spokane",
        "stlouis": "St. Louis",
        "stockton": "Stockton",
        "stpetersburg": "St. Petersburg",
        "tampa": "Tampa",
        "toledo": "Toledo",
        "tucson": "Tucson",
        "tulsa": "Tulsa",
        "virginiabeach": "Virginia Beach",
        "washingtondc": "Washington DC",
        "wichita": "Wichita",
        "winstonsalem": "Winston Salem"
    }
    
    categories = [
        "affect",
        "posemo",
        "negemo",
        "anx",
        "anger",
        "sad",
        "swear",
        "achieve",
        "social",
        "we",
        "family",
        # "friend",
        "cause",
        "tentat",
        "certain",
        "insight",
        "health",
        "ingest",
        "bio",
        "body",
        "motion",
        "space",
        "time",
        "home",
        "work",
        "money"
    ]

    universal_traj_categories = [
        "anx",
        "health",
        "posemo",
        "bio",
        "achieve",
        "tentat",
        "body",
        "certain",
        "sad",
        "affect",
        "social",
        "work",
        "money",
        "family",
        "insight",
        "motion",
        "space"
    ]

      # for category in categories:
    
    # plot_city_comparison(
    #     f'../metrics/albuquerque_metrics.parquet',
    #     f'../metrics/anchorage_metrics.parquet',
    #     city1='Albuquerque',
    #     city2='Anchorage',
    #     metric='Raw Number of Posts',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/albuquerque_anchorage_post_count.png',
    # )

    # plot_city_comparison(
    #     f'../metrics/coloradosprings_metrics.parquet',
    #     f'../metrics/bakersfield_metrics.parquet',
    #     city1='Colorado Springs',
    #     city2='Bakersfield',
    #     metric='Raw Number of Comments',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/coloradosprings_baskersfield_comment_count.png',
    # )

    # plot_city_comparison(
    #     f'../metrics/seattle_metrics.parquet',
    #     f'../metrics/atlanta_metrics.parquet',
    #     city1='Seattle',
    #     city2='Atlanta',
    #     metric='Percentage of Posts with Comments',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/seattle_atlanta_comment_percentage.png',
    # )

    # plot_response_time_comparison(
    #     '../metrics/charlotte_metrics.parquet',
    #     '../metrics/baltimore_metrics.parquet',
    #     city1='Charlotte',
    #     city2='Baltimore',
    #     resample_unit='W',
    #     metric='Average Response Time with Cutoff (Minutes)',
    #     time_diff_unit='minutes',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/charlotte_baltimore_response_time.png',
    # )

    # plot_post_lifespan_three_way(
    #     '../metrics/boise_metrics.parquet',
    #     '../metrics/boston_metrics.parquet',
    #     '../metrics/cincinnati_metrics.parquet',
    #     city1='Boise',
    #     city2='Boston',
    #     city3='Cincinnati',
    #     resample_unit='weekly',
    #     metric='Post Lifespan (Mean in hours)',
    #     units='hours',
    #     statistic='mean',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/boise_boston_cincinnati_lifespan.png'
    # )

    # plot_city_comparison(
    #     '../metrics/losangeles_metrics.parquet',
    #     '../metrics/lasvegas_metrics.parquet',
    #     city1='Los Angeles',
    #     city2='Las Vegas',
    #     metric='Average Sentiment',
    #     save_plot=True,
    #     plot_path='../city_comparison_graphs/losangeles_lasvegas_average_sentiment.png',
    # )


    # metric = 'Average Response Time with Cutoff (Minutes)',
    # metric='Post Lifespan (Mean in hours)',
    # metric='Average Sentiment',
    # metric='Positive Sentiment Count',
    # metric='Negative Sentiment Count'

    #     plot_liwc_clusters_timeseries(
    #         f"../liwc_cluster_aggregated_metrics_normalised_smoothed/aggregated_{category}.parquet", 
    #         f"Clustered Intensities of LIWC {category} Category", "Intensity", 
    #         "2019-10-01",
    #         "2022-12-31",
    #         f"../liwc_cluster_graphs_normalised_smoothed_bigger_font/{category}_clusters.png"
    #     )

    # for category in universal_traj_categories:
    #     plot_liwc_clusters_timeseries(
    #         f"../liwc_universal_aggregated_metrics/aggregated_{category}.parquet", 
    #         f"Intensities of LIWC {category} Category", "Intensity", 
    #         "2019-10-01",
    #         "2022-12-31",
    #         f"../liwc_universal_graphs_bigger_font/{category}_universal.png"
    #     )

    for category in universal_traj_categories:
        print(category)
        plot_word_shifts_from_parquet(
            filepath=f'../liwc_word_shifts_normalised_universal/{category}_word_shifts.parquet',
            category=category,
            top_n=20,
            save=True,
            filename=f'../word_shifts_20/{category}_word_shifts.png'
        )
    
    # generate_graphs_for_cities(city_dict)

    # plot_us_cities_map_cartopy(
    #     city_dict,
    #     annotate=False,
    #     save_path="../us_city_map.png"
    # )

    # plot_us_cities_map_from_dict(
    #     city_dict,
    #     annotate=False,
    #     save_path="../final_us_city_map.png"
    # )

    # plot_shape_vs_scale("../shape_scale.parquet", save_path="../shape_scale.png")
    
    # plot_shape_vs_scale_knn("../shape_scale/shape_scale.parquet", "../shape_scale_knn3.png", 3)
    # plot_shape_vs_scale_knn("../structural_metrics_shape_scale.parquet", "../structural_metrics_shape_scale_knn3.png", 3)
