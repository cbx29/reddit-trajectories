#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from math import sqrt
import statsmodels.api as sm
import json
import scipy.stats as stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from typing import Dict, Set, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def num_lines(file):
    with open(file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return len(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {file}: {e}")
            return 0


def lowest_entries_per_week(json_file, output_dir, dataset_type):
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

    print(min(weekly_counts))


def count_unique_authors(file_path):
    """
    Counts the number of unique authors in a Reddit submissions or comments JSON array file,
    ignoring entries where the author is "[deleted]".
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return 0

    unique_authors = set()
    for entry in data:
        author = entry.get('author')
        if author and author != '[deleted]':
            unique_authors.add(author)

    return len(unique_authors)


def print_bot_authors(input_file):
    with open(input_file, "r", encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {input_file}: {e}")
            return False

    if not data:
        print(f"No data to filter in file: {input_file}")
        return False
    
    bot_authors = set()
    print("Bots:")
    for entry in data:
        author = entry.get('author', '')
        print(author.lower())
        if "bot" in author.lower():
            bot_authors.add(author)
    for author in sorted(bot_authors):
        print(author)


def calculate_spearman_correlation(cities, city_populations, type):
    """
    Calculate the Spearman correlation between city population size and subreddit activity.

    Parameters:
        parquet_path (str): Path to the parquet file containing comments or submissions data.
        city_population_dict (dict): Dictionary with city names as keys and population sizes as values.

    Returns:
        None: Prints the Spearman correlation value.
    """

    activities = []
    populations = []

    for city in cities:
        data = pd.read_parquet(f"../data/{city}/{type}/processed/{type}.parquet")

        activity_count = len(data)
        activities.append(activity_count)
        population = city_populations[city]
        populations.append(population)

    df = pd.DataFrame({
        'Activity': activities,
        "Population": populations
    })

    spearman_corr, p_value = spearmanr(df["Population"], df["Activity"])
    
    print(spearman_corr, p_value)


def count_unique_authors_parquet(parquet_file):
    """
    Counts the number of unique authors in a Parquet file using the 'author' field.

    Parameters:
    parquet_file (str): The file path to the Parquet file.

    Returns:
    int: The number of unique authors.
    """
    df = pd.read_parquet(parquet_file)

    if 'author' not in df.columns:
        raise ValueError("The 'author' field is not present in the Parquet file.")

    unique_authors_count = df['author'].nunique()

    return unique_authors_count


def user_population_spearman(city_users, city_populations):
    common_cities = set(city_populations.keys()) & set(city_users.keys())

    missing_in_users = set(city_populations.keys()) - set(city_users.keys())
    missing_in_populations = set(city_users.keys()) - set(city_populations.keys())

    if missing_in_users:
        print(f"Cities missing in users data: {missing_in_users}")
    if missing_in_populations:
        print(f"Cities missing in population data: {missing_in_populations}")

    populations = []
    users = []

    for city in common_cities:
        populations.append(city_populations[city])
        users.append(city_users[city])

    spearman_corr, p_value = stats.spearmanr(populations, users)

    print(f"Spearman Correlation Coefficient: {spearman_corr}")
    print(f"P-value: {p_value}")


def compute_new_and_returning_users(
    pre_covid_folder: str,
    during_covid_folder: str,
    output_path: Optional[str] = None,
    return_df: bool = False
) -> Optional[pd.DataFrame]:
    """
    Computes the number of new and returning users for each city subreddit
    by comparing pre-COVID and during-COVID user activity. It can save the results
    to a Parquet file, plot the data, or return the resulting DataFrame based
    on the provided parameters.

    Parameters:
    - pre_covid_folder (str): Path to the folder containing pre-COVID parquet files.
    - during_covid_folder (str): Path to the folder containing during-COVID parquet files.
    - output_path (Optional[str]): Path where the resulting Parquet file will be saved.
                                    If None, the file will not be saved. Default is None.
    - return_df (bool): If True, returns the DataFrame. If False, plots the data. Default is False.

    Returns:
    - pd.DataFrame or None: A DataFrame with columns ['City', 'New_Users', 'Returning_Users', 
                            'Example_New_User', 'Example_Returning_User'] if return_df is True.
                            Otherwise, None.
    """
    
    def extract_city_names(folder: str) -> Set[str]:
        city_names = set()
        for file in os.listdir(folder):
            if file.endswith('.parquet'):
                city = file.rsplit('_', 1)[0]
                city_names.add(city)
        return city_names

    pre_cities = extract_city_names(pre_covid_folder)
    during_cities = extract_city_names(during_covid_folder)
    
    all_cities = pre_cities.union(during_cities)
    
    results = []

    for city in sorted(all_cities):
        pre_comments_path = os.path.join(pre_covid_folder, f"{city}_comments.parquet")
        pre_submissions_path = os.path.join(pre_covid_folder, f"{city}_submissions.parquet")
        during_comments_path = os.path.join(during_covid_folder, f"{city}_comments.parquet")
        during_submissions_path = os.path.join(during_covid_folder, f"{city}_submissions.parquet")
        
        pre_authors = set()
        during_authors = set()
        
        if os.path.exists(pre_comments_path):
            try:
                df_pre_comments = pd.read_parquet(pre_comments_path, columns=['author'])
                pre_authors.update(df_pre_comments['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {pre_comments_path}: {e}")
        
        if os.path.exists(pre_submissions_path):
            try:
                df_pre_submissions = pd.read_parquet(pre_submissions_path, columns=['author'])
                pre_authors.update(df_pre_submissions['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {pre_submissions_path}: {e}")
        
        if os.path.exists(during_comments_path):
            try:
                df_during_comments = pd.read_parquet(during_comments_path, columns=['author'])
                during_authors.update(df_during_comments['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {during_comments_path}: {e}")
        
        if os.path.exists(during_submissions_path):
            try:
                df_during_submissions = pd.read_parquet(during_submissions_path, columns=['author'])
                during_authors.update(df_during_submissions['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {during_submissions_path}: {e}")
        
        new_users = during_authors - pre_authors
        returning_users = during_authors.intersection(pre_authors)
        
        example_new_user: Optional[str] = next(iter(new_users), None) if new_users else None
        example_returning_user: Optional[str] = next(iter(returning_users), None) if returning_users else None
        
        results.append({
            'City': city,
            'New_Users': len(new_users),
            'Returning_Users': len(returning_users),
            'Example_New_User': example_new_user,
            'Example_Returning_User': example_returning_user
        })
    
    results_df = pd.DataFrame(results)
    
    if output_path:
        try:
            results_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
            print(f"User statistics saved to {output_path}")
        except Exception as e:
            print(f"Error saving to Parquet file: {e}")
    
    if return_df:
        return results_df
    else:
        try:
            plt.figure(figsize=(16, 10))
            bar_width = 0.35
            indices = range(len(results_df))
            
            plt.bar(
                [i - bar_width/2 for i in indices],
                results_df['New_Users'],
                width=bar_width,
                label='New Users',
                color='skyblue'
            )
            plt.bar(
                [i + bar_width/2 for i in indices],
                results_df['Returning_Users'],
                width=bar_width,
                label='Returning Users',
                color='salmon'
            )
            
            plt.xlabel('City', fontsize=14)
            plt.ylabel('Number of Users', fontsize=14)
            plt.title('New and Returning Users per City Subreddit', fontsize=16)
            plt.xticks(indices, results_df['City'], rotation=90, ha='center')
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.title('New and Returning Users per City Subreddit', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error during plotting: {e}")
        return None


def calculate_and_plot_post_count(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W',
    save_plot=False,
    plot_path='post_count_timeseries.png',
    display_plot=True
):
    """
    Plots the time series for the raw number of posts over time by combining pre-COVID
    and main (COVID) datasets.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing at least 'id' and 'created_utc'.
    comments_file : str
        (Not used in this version; kept for consistency.)
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        (Not used in this version; kept for consistency.)
    city : str, optional
        Name of the city for plot titling.
    resample_unit : str, optional
        Pandas resample frequency (default is 'W' for weekly).
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
    import pandas as pd
    import matplotlib.pyplot as plt

    def process_data(posts_file):
        print(f"Loading posts data from {posts_file}...")
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")

        print("Converting timestamps...")
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')

        posts.sort_values('created_datetime', inplace=True)

        posts.set_index('created_datetime', inplace=True)

        print(f"Resampling by '{resample_unit}' to compute the number of posts...")
        post_counts = posts.resample(resample_unit).size()
        return post_counts

    print("Processing pre-COVID posts data...")
    precovid_counts = process_data(precovid_posts_file)
    precovid_counts = precovid_counts['2019-10-01':'2019-12-31']

    print("Processing main posts data...")
    main_counts = process_data(posts_file)

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


def calculate_and_plot_comment_count(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W',
    save_plot=False,
    plot_path='comment_count_timeseries.png',
    display_plot=True
):
    """
    Plots the time series for the raw number of comments over time by combining pre-COVID
    and main (COVID) datasets.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file.
        (Not used in this version; retained for interface consistency.)
    comments_file : str
        Path to the main comments Parquet file containing 'parent_id' and 'created_utc'.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
        (Not used in this version; retained for interface consistency.)
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        Name of the city (if applicable) to be displayed in the plot title.
    resample_unit : str, optional
        Pandas resample frequency (default is 'W' for weekly).
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
    import pandas as pd
    import matplotlib.pyplot as plt

    def process_data(comments_file):
        print(f"Loading comments data from {comments_file}...")
        comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")

        print("Converting timestamps...")
        comments['created_datetime'] = pd.to_datetime(comments['created_utc'], unit='s')

        comments.sort_values('created_datetime', inplace=True)

        comments.set_index('created_datetime', inplace=True)

        print(f"Resampling by '{resample_unit}' to compute the number of comments...")
        comment_counts = comments.resample(resample_unit).size()
        return comment_counts
    
    print("Processing pre-COVID comments data...")
    precovid_counts = process_data(precovid_comments_file)
    precovid_counts = precovid_counts['2019-10-01':'2019-12-31']

    print("Processing main comments data...")
    main_counts = process_data(comments_file)

    print("Combining datasets...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', label='Number of Comments', color='blue')

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


def calculate_and_plot_post_lifespan_timeseries(
    posts_parquet_path,
    comments_parquet_path,
    aggregation='monthly',
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
    Calculates and plots the lifespan of posts over time, aggregated by a specified interval.
    
    Parameters:
    - posts_parquet_path (str): Path to the posts Parquet file containing 'id' and 'created_utc' fields.
    - comments_parquet_path (str): Path to the comments Parquet file containing 'parent_id' and 'created_utc' fields.
    - aggregation (str): Time interval for aggregation. Options: 'daily', 'weekly', 'monthly', 'yearly'. Default is 'monthly'.
    - units (str): Time units for lifespan. Options: 'seconds', 'minutes', 'hours', 'days'. Default is 'hours'.
    - statistic (str): Statistic to plot. Options: 'mean', 'median', 'count'. Default is 'mean'.
    - title (str): Title of the plot. Default is 'Post Lifespan Over Time'.
    - xlabel (str): Label for the x-axis. If None, it will be set based on aggregation.
    - ylabel (str): Label for the y-axis. If None, it will be set based on statistic and units.
    - figsize (tuple): Size of the plot figure. Default is (14, 7).
    - return_data (bool): If True, returns a DataFrame with aggregated lifespan statistics. Default is False.
    - log_scale (bool): If True, applies logarithmic scaling to the y-axis. Default is False.
    - verbose (bool): If True, prints detailed logs for debugging purposes. Default is False.
    - save_plot (bool): If True, saves the generated plot to a file.
    - plot_path (str): File path to save the plot if save_plot is True.
    
    Returns:
    - If return_data=True, returns a pandas DataFrame with aggregated statistics.
    - Otherwise, returns None.
    """
    
    valid_aggregations = ['daily', 'weekly', 'monthly', 'yearly']
    if aggregation not in valid_aggregations:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from {valid_aggregations}.")
    
    valid_units = ['seconds', 'minutes', 'hours', 'days']
    if units not in valid_units:
        raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")
    
    valid_statistics = ['mean', 'median', 'count']
    if statistic not in valid_statistics:
        raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")
    
    if xlabel is None:
        xlabel = f'Time ({aggregation.capitalize()})'
    if ylabel is None:
        if statistic in ['mean', 'median']:
            ylabel = f'Post Lifespan ({units.capitalize()})'
        elif statistic == 'count':
            ylabel = 'Number of Posts'
    
    if verbose:
        print("Starting plot_post_lifespan_timeseries function...")
        print(f"Aggregation: {aggregation}, Units: {units}, Statistic: {statistic}")
    
    try:
        if verbose:
            print(f"Reading posts data from {posts_parquet_path}...")
        posts_df = pd.read_parquet(posts_parquet_path, columns=['id', 'created_utc'])
    except Exception as e:
        print(f"Error reading the posts Parquet file: {e}")
        return
    
    if 'id' not in posts_df.columns or 'created_utc' not in posts_df.columns:
        print("The posts Parquet file must contain 'id' and 'created_utc' columns.")
        return
    
    if verbose:
        print(f"Number of posts loaded: {len(posts_df)}")
    
    if not np.issubdtype(posts_df['created_utc'].dtype, np.number):
        try:
            posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
            if verbose:
                print("'created_utc' in posts converted to datetime.")
        except Exception as e:
            print(f"Error converting 'created_utc' in posts to datetime: {e}")
            return
    else:
        posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
        if verbose:
            print("'created_utc' in posts converted from Unix timestamp to datetime.")
    
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
        try:
            comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
            if verbose:
                print("'created_utc' in comments converted to datetime.")
        except Exception as e:
            print(f"Error converting 'created_utc' in comments to datetime: {e}")
            return
    else:
        comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
        if verbose:
            print("'created_utc' in comments converted from Unix timestamp to datetime.")
    
    if verbose:
        print("Merging comments with posts...")
    merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner', suffixes=('_comment', '_post'))
    
    if verbose:
        print(f"Number of comments after merging with posts: {len(merged_df)}")
        print("Sample of merged data:")
        print(merged_df.head())
    
    if verbose:
        print("Calculating min and max 'created_utc' for each post...")
    lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
    lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

    if verbose:
        print(f"Number of posts with comments: {len(lifespan_df)}")
        print("Sample of lifespan data:")
        print(lifespan_df.head())
    
    lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()
    
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
        print("No posts with lifespan greater than 0. Cannot plot timeseries.")
        return
    
    if verbose:
        print("Merging lifespan data with post creation times...")
    lifespan_df = lifespan_df.merge(posts_df[['id', 'created_utc']], left_on='post_id', right_on='id', how='left')
    lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
    lifespan_df.drop('id', axis=1, inplace=True)
    
    if verbose:
        print(f"Aggregating lifespans by {aggregation}...")
    lifespan_df.set_index('post_created_time', inplace=True)
    
    if aggregation == 'daily':
        resample_rule = 'D'
    elif aggregation == 'weekly':
        resample_rule = 'W'
    elif aggregation == 'monthly':
        resample_rule = 'M'
    elif aggregation == 'yearly':
        resample_rule = 'Y'
    
    if statistic in ['mean', 'median']:
        aggregated = lifespan_df['lifespan'].resample(resample_rule).agg(statistic)
    elif statistic == 'count':
        aggregated = lifespan_df['lifespan'].resample(resample_rule).count()
    
    if verbose:
        print("Aggregated lifespan statistics:")
        print(aggregated.head())
    
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")
    
    if statistic in ['mean', 'median']:
        sns.lineplot(x=aggregated.index, y=aggregated.values, marker='o', color='skyblue')
    elif statistic == 'count':
        sns.barplot(x=aggregated.index, y=aggregated.values, color='skyblue')
    
    if log_scale:
        plt.yscale('log')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    if aggregation in ['monthly', 'yearly']:
        plt.xticks(rotation=45)
    
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
        aggregated_df = aggregated.reset_index()
        aggregated_df.columns = ['Time', f'Lifespan_{statistic}_{units}']
        return aggregated_df
    

def calculate_and_plot_response_times(posts_file, comments_file, city='', time_unit='W', time_diff_unit='minutes', save_plot=False, plot_path='average_response_times.png', display_plot=True):
    """
    Plots the average response times of city subreddit posts.

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - comments_file (str): Path to the comments Parquet file.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - time_diff_unit (str): Unit for response time ('seconds', 'minutes', 'hours', 'days').
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.

    Returns:
    - None: Displays and optionally saves a plot of average response times.
    """
    print("Reading posts data...")
    posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
    print(f"Number of posts: {len(posts)}")

    print("Reading comments data...")
    comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
    print(f"Number of comments: {len(comments)}")

    print("Renaming 'created_utc' columns...")
    posts = posts.rename(columns={'created_utc': 'created_utc_post'})
    comments = comments.rename(columns={'created_utc': 'created_utc_comment'})

    print("Preparing posts data...")
    posts['parent_id'] = 't3_' + posts['id']
    posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')

    print("Merging comments with posts...")
    merged = comments.merge(
        posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
        on='parent_id',
        how='inner'  # Change to 'left' if you want to include posts without comments
    )
    print(f"Merged DataFrame size: {merged.shape}")

    print("Identifying first comments for each post...")
    merged_sorted = merged.sort_values(['parent_id', 'created_utc_comment'])
    first_comments = merged_sorted.groupby('parent_id').first().reset_index()
    print(f"Number of posts with at least one comment: {len(first_comments)}")

    print("Calculating response times...")
    first_comments['response_time_seconds'] = first_comments['created_utc_comment'] - first_comments['created_utc_post']

    if time_diff_unit == 'seconds':
        first_comments['response_time'] = first_comments['response_time_seconds']
        ylabel = 'Average Response Time (Seconds)'
    elif time_diff_unit == 'minutes':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 60
        ylabel = 'Average Response Time (Minutes)'
    elif time_diff_unit == 'hours':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 3600
        ylabel = 'Average Response Time (Hours)'
    elif time_diff_unit == 'days':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 86400
        ylabel = 'Average Response Time (Days)'
    else:
        raise ValueError("Unsupported time_diff_unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")

    first_comments['created_datetime_post'] = pd.to_datetime(first_comments['created_datetime_post'])
    first_comments.set_index('created_datetime_post', inplace=True)

    print(f"Resampling data with time unit: {time_unit}")
    avg_response = first_comments['response_time'].resample(time_unit).mean()

    print("Plotting the average response times...")
    plt.figure(figsize=(12, 6))
    avg_response.plot(marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(f'{city} Average Response Times Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def calculate_and_plot_response_times_with_cutoff(
    posts_file,
    comments_file,
    city='',
    time_unit='W',
    time_diff_unit='minutes',
    save_plot=False,
    plot_path='average_response_times.png',
    display_plot=True
):
    """
    Plots the average response times of city subreddit posts, excluding responses
    that occur more than 24 hours after a post is made.

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - comments_file (str): Path to the comments Parquet file.
    - city (str): City name for the plot title.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - time_diff_unit (str): Unit for response time ('seconds', 'minutes', 'hours', 'days').
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.
    - display_plot (bool): Whether to display the plot.

    Returns:
    - None: Displays and optionally saves a plot of average response times.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    print("Reading posts data...")
    posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
    print(f"Number of posts: {len(posts)}")

    print("Reading comments data...")
    comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
    print(f"Number of comments: {len(comments)}")

    print("Renaming 'created_utc' columns...")
    posts = posts.rename(columns={'created_utc': 'created_utc_post'})
    comments = comments.rename(columns={'created_utc': 'created_utc_comment'})

    print("Preparing posts data...")
    posts['parent_id'] = 't3_' + posts['id']
    posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')

    print("Merging comments with posts...")
    merged = comments.merge(
        posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
        on='parent_id',
        how='inner'
    )
    print(f"Merged DataFrame size: {merged.shape}")

    print("Identifying first comments for each post...")
    merged_sorted = merged.sort_values(['parent_id', 'created_utc_comment'])
    first_comments = merged_sorted.groupby('parent_id').first().reset_index()
    print(f"Number of posts with at least one comment: {len(first_comments)}")

    print("Calculating response times...")
    first_comments['response_time_seconds'] = (
        first_comments['created_utc_comment'] - first_comments['created_utc_post']
    )

    first_comments = first_comments[first_comments['response_time_seconds'] <= 86400]

    if time_diff_unit == 'seconds':
        first_comments['response_time'] = first_comments['response_time_seconds']
        ylabel = 'Average Response Time (Seconds)'
    elif time_diff_unit == 'minutes':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 60
        ylabel = 'Average Response Time (Minutes)'
    elif time_diff_unit == 'hours':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 3600
        ylabel = 'Average Response Time (Hours)'
    elif time_diff_unit == 'days':
        first_comments['response_time'] = first_comments['response_time_seconds'] / 86400
        ylabel = 'Average Response Time (Days)'
    else:
        raise ValueError("Unsupported time_diff_unit. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

    first_comments['created_datetime_post'] = pd.to_datetime(first_comments['created_datetime_post'])
    first_comments.set_index('created_datetime_post', inplace=True)

    print(f"Resampling data with time unit: {time_unit}")
    avg_response = first_comments['response_time'].resample(time_unit).mean()

    print("Plotting the average response times with cutoff...")
    plt.figure(figsize=(12, 6))
    avg_response.plot(marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(f'{city} Average Response Times (Responses within 24 Hours) Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display_plot:
        plt.show()
    else:
        plt.close()


def calculate_and_plot_comment_percentage_time_window(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W',
    save_plot=False,
    plot_path='comment_percentage_7day.png',
    display_plot=True
):

    def process_data(posts_file, comments_file):
        print(f"Loading posts data from {posts_file}...")
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")

        print(f"Loading comments data from {comments_file}...")
        comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")

        print("Converting timestamps...")
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc'], unit='s')

        posts['parent_id'] = 't3_' + posts['id']

        posts.sort_values('created_datetime_post', inplace=True)
        comments.sort_values('created_datetime_comment', inplace=True)

        comment_groups = comments.groupby('parent_id')['created_datetime_comment']

        print("Checking comments in [post_time, post_time + 7 days) for each post...")

        def has_comment_in_7day_window(row):
            pid = row['parent_id']
            post_time = row['created_datetime_post']
            window_end = post_time + pd.Timedelta(days=7)

            if pid not in comment_groups.groups:
                return False
            c_times = comment_groups.get_group(pid)

            in_window = c_times[(c_times >= post_time) & (c_times < window_end)]
            return len(in_window) > 0

        posts['has_comment_7day'] = posts.apply(has_comment_in_7day_window, axis=1)

        print(f"Resampling by '{resample_unit}' to compute the percentage of posts with 7-day comments...")
        posts.set_index('created_datetime_post', inplace=True)
        total_posts = posts.resample(resample_unit).size()
        commented_posts = posts[posts['has_comment_7day']].resample(resample_unit).size()
        percentage_with_comments = (commented_posts / total_posts) * 100

        return percentage_with_comments

    print("Processing data...")
    precovid_percentage = process_data(precovid_posts_file, precovid_comments_file)
    precovid_percentage = precovid_percentage['2019-10-01':'2019-12-31']
    main_percentage = process_data(posts_file, comments_file)

    print("Combining datasets...")
    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_percentage.plot(marker='o', linestyle='-', label='Percentage of Posts with Comments', color='blue')

    covid_start_date = pd.Timestamp('2020-01-01')
    plt.axvline(covid_start_date, color='red', linestyle='--', label='COVID Start Date (Jan 2020)')

    plt.xlabel('Time')
    plt.ylabel('Percentage of Posts with Comments Within 7 Days (%)')
    plt.title(f'{city} Percentage of Posts Receiving Comments Within 7 Days')
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


def calculate_and_plot_post_sentiment(posts_file, city='', text_column='selftext', time_unit='W', save_plot=False, plot_path='post_sentiment.png', display_plot=True):
    """
    Analyzes and plots the average sentiment of Reddit posts over time using VADER.

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - text_column (str): Column name in posts_file that contains the text to analyze.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.

    Returns:
    - None: Displays and optionally saves a plot of average sentiment over time.
    """

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    print("Computing sentiment scores...")
    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    print(f"Resampling data with time unit: {time_unit}")
    avg_sentiment = posts['sentiment'].resample(time_unit).mean()

    print("Plotting the average sentiment over time...")
    plt.figure(figsize=(12, 6))
    avg_sentiment.plot(marker='o', linestyle='-')
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


def calculate_and_plot_negative_sentiment_count(
        posts_file,
        city='',
        text_column='selftext',
        time_unit='W',
        threshold=-0.05,
        save_plot=False,
        plot_path='negative_sentiment_count.png',
        display_plot=True
    ):
    """
    Analyzes and plots the number of Reddit posts with sentiment below a given threshold 
    (e.g., 0.05) over time using VADER.

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - city (str): Name or label of the city (or context) for the plot title.
    - text_column (str): Column name in posts_file that contains the text to analyze.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - threshold (float): Sentiment threshold below which a post is counted as "negative".
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.
    - display_plot (bool): Whether to display the plot.

    Returns:
    - None: Displays and optionally saves a plot of the negative-sentiment post count over time.
    """

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    print("Computing sentiment scores...")

    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    print("Identifying negative posts and resampling data...")
    posts['is_negative'] = (posts['sentiment'] < threshold).astype(int)
    
    negative_post_count = posts['is_negative'].resample(time_unit).sum()

    print(f"Plotting the number of posts with sentiment < {threshold} over time...")
    plt.figure(figsize=(12, 6))
    negative_post_count.plot(marker='o', linestyle='-')
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


def calculate_and_plot_positive_sentiment_count(
        posts_file,
        city='',
        text_column='selftext',
        time_unit='W',              
        threshold=0.05,          
        save_plot=False,
        plot_path='positive_sentiment_count.png',
        display_plot=True
    ):
    """
    Analyzes and plots the number of Reddit posts with sentiment above a given threshold 
    (e.g., 0.05) over time using VADER.

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - city (str): Name or label of the city (or context) for the plot title.
    - text_column (str): Column name in posts_file that contains the text to analyze.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - threshold (float): Sentiment threshold above which a post is counted as "positive".
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.
    - display_plot (bool): Whether to display the plot.

    Returns:
    - None: Displays and optionally saves a plot of the positive-sentiment post count over time.
    """
   
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    print("Computing sentiment scores...")
    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    print("Identifying positive posts and resampling data...")
    posts['is_positive'] = (posts['sentiment'] > threshold).astype(int)

    positive_post_count = posts['is_positive'].resample(time_unit).sum()

    print(f"Plotting the number of posts with sentiment > {threshold} over time...")
    plt.figure(figsize=(12, 6))
    positive_post_count.plot(marker='o', linestyle='-')
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


def get_raw_post_count_metrics(
    posts_file,
    comments_file,  
    precovid_posts_file,
    precovid_comments_file, 
    city='',
    resample_unit='W'  
):
    """
    Calculates and returns the raw number of posts metrics for both pre-COVID and main datasets.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing 'id' and 'created_utc'.
    comments_file : str
        (Not used in this version, kept for consistency.)
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        (Not used in this version, kept for consistency.)
    city : str, optional
        (Not used in this version.)
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).

    Returns
    -------
    pd.DataFrame
        A DataFrame where the row is labeled 'Raw Number of Posts'
        and columns are date strings (DD/MM/YYYY).
    """
    import pandas as pd

    def process_posts(posts_path):
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc'])

        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')

        posts.sort_values('created_datetime', inplace=True)

        posts.set_index('created_datetime', inplace=True)

        post_counts = posts.resample(resample_unit).size()
        return post_counts

    precovid_counts = process_posts(precovid_posts_file)
    precovid_counts = precovid_counts['2016-01-01':'2019-12-31']

    main_counts = process_posts(posts_file)

    combined_counts = pd.concat([precovid_counts, main_counts])

    full_index = pd.date_range(
        start=combined_counts.index.min(),
        end=combined_counts.index.max(),
        freq=resample_unit
    )
    combined_counts = combined_counts.reindex(full_index).fillna(0)

    metrics_table = pd.DataFrame(
        {'Raw Number of Posts': combined_counts.values},
        index=combined_counts.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_raw_comment_count_metrics(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W'
):
    """
    Calculates and returns the raw number of comments metrics for both pre-COVID and main datasets.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file.
        (Not used in this version, retained for interface consistency.)
    comments_file : str
        Path to the main comments Parquet file containing 'parent_id' and 'created_utc'.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
        (Not used in this version, retained for interface consistency.)
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        (Not used in this version.)
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).

    Returns
    -------
    pd.DataFrame
        A DataFrame where the row is labeled 'Raw Number of Comments'
        and the columns are date strings in the format 'DD/MM/YYYY'.
    """
    import pandas as pd

    def process_comments(comments_path):
        comments = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        comments['created_datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
        comments.sort_values('created_datetime', inplace=True)
        comments.set_index('created_datetime', inplace=True)
        comment_counts = comments.resample(resample_unit).size()
        return comment_counts

    precovid_counts = process_comments(precovid_comments_file)
    precovid_counts = precovid_counts['2016-01-01':'2019-12-31']

    main_counts = process_comments(comments_file)

    combined_counts = pd.concat([precovid_counts, main_counts])

    full_index = pd.date_range(
        start=combined_counts.index.min(),
        end=combined_counts.index.max(),
        freq=resample_unit
    )
    combined_counts = combined_counts.reindex(full_index).fillna(0)

    metrics_table = pd.DataFrame(
        {'Raw Number of Comments': combined_counts.values},
        index=combined_counts.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_comment_percentage_metrics(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W'
):
    """
    Calculates and returns the comment percentage metrics for both pre-COVID and main datasets.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing 'id' and 'created_utc'.
    comments_file : str
        Path to the main comments Parquet file containing 'parent_id' and 'created_utc'.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        (Not used in this version, kept for consistency if needed.)
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).

    Returns
    -------
    pd.DataFrame
        A DataFrame where the row is labeled 'Percentage of Posts with Comments'
        and columns are date strings (DD/MM/YYYY). 
    """
    import pandas as pd

    def process_data(posts_path, comments_path):
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc'])
        comments = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc'], unit='s')

        posts['parent_id'] = 't3_' + posts['id']

        posts.sort_values('created_datetime_post', inplace=True)
        comments.sort_values('created_datetime_comment', inplace=True)

        comment_groups = comments.groupby('parent_id')['created_datetime_comment']

        def has_comment_in_7day_window(row):
            pid = row['parent_id']
            post_time = row['created_datetime_post']
            window_end = post_time + pd.Timedelta(days=7)

            if pid not in comment_groups.groups:
                return False
            c_times = comment_groups.get_group(pid)

            in_window = c_times[(c_times >= post_time) & (c_times < window_end)]
            return len(in_window) > 0

        posts['has_comment_7day'] = posts.apply(has_comment_in_7day_window, axis=1)

        posts.set_index('created_datetime_post', inplace=True)

        total_posts = posts.resample(resample_unit).size()
        commented_posts = posts[posts['has_comment_7day']].resample(resample_unit).size()
        percentage_with_comments = (commented_posts / total_posts) * 100

        return percentage_with_comments

    precovid_percentage = process_data(precovid_posts_file, precovid_comments_file)
    precovid_percentage = precovid_percentage['2016-01-01':'2019-12-31']

    main_percentage = process_data(posts_file, comments_file)

    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    full_index = pd.date_range(
        start=combined_percentage.index.min(),
        end=combined_percentage.index.max(),
        freq=resample_unit
    )
    combined_percentage = combined_percentage.reindex(full_index)

    combined_percentage = combined_percentage.interpolate()

    metrics_table = pd.DataFrame(
        {'Percentage of Posts with Comments': combined_percentage.values},
        index=combined_percentage.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_post_lifespan_timeseries(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    aggregation='weekly',
    units='hours',
    statistic='mean'
):
    """
    Calculates the post-lifespan time series for both pre-COVID and main datasets,
    combines them into a single DataFrame, and returns that DataFrame without saving to disk.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing 'id' and 'created_utc'.
    comments_file : str
        Path to the main comments Parquet file containing 'parent_id' and 'created_utc'.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file (e.g., 2019-07-01 to 2019-12-31).
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        (Not strictly used in this version; kept for consistency if needed.)
    aggregation : str
        Time interval for aggregation. One of ['daily', 'weekly', 'monthly', 'yearly'].
    units : str
        Time units for lifespan: 'seconds', 'minutes', 'hours', or 'days'.
    statistic : str
        Statistic to compute: 'mean', 'median', or 'count'.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame where:
          - row index is the statistic name (e.g., "Post Lifespan (Mean in hours)"),
          - column indices are string-formatted dates (DD/MM/YYYY),
          - values are the aggregated lifespan (or count).
    """
    import numpy as np
    import pandas as pd
    
    def process_data(posts_path, comments_path, aggregation, units, statistic):
        """Process posts and comments to compute the aggregated post-lifespan time series."""

        posts_df = pd.read_parquet(posts_path, columns=['id', 'created_utc'])
        posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')

        comments_df = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)
        comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')

        merged_df = comments_df.merge(
            posts_df,
            left_on='parent_id_clean',
            right_on='id',
            how='inner',
            suffixes=('_comment', '_post')
        )

        lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
        lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

        lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()

        lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

        if units == 'seconds':
            lifespan_df['lifespan'] = lifespan_df['lifespan_seconds']
        elif units == 'minutes':
            lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 60
        elif units == 'hours':
            lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 3600
        elif units == 'days':
            lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 86400
        else:
            raise ValueError(f"Invalid units '{units}'. Choose from ['seconds', 'minutes', 'hours', 'days'].")

        lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]

        lifespan_df = lifespan_df.merge(
            posts_df[['id', 'created_utc']],
            left_on='post_id',
            right_on='id',
            how='left'
        )
        lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
        lifespan_df.set_index('post_created_time', inplace=True)
        lifespan_df.drop('id', axis=1, inplace=True)

        valid_aggregations = ['daily', 'weekly', 'monthly', 'yearly']
        if aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from {valid_aggregations}.")

        if aggregation == 'daily':
            resample_rule = 'D'
        elif aggregation == 'weekly':
            resample_rule = 'W'
        elif aggregation == 'monthly':
            resample_rule = 'M'
        elif aggregation == 'yearly':
            resample_rule = 'Y'

        valid_statistics = ['mean', 'median', 'count']
        if statistic not in valid_statistics:
            raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")

        if statistic in ['mean', 'median']:
            aggregated_series = lifespan_df['lifespan'].resample(resample_rule).agg(statistic)
        else:
            aggregated_series = lifespan_df['lifespan'].resample(resample_rule).count()

        return aggregated_series

    precovid_lifespan = process_data(
        precovid_posts_file,
        precovid_comments_file,
        aggregation=aggregation,
        units=units,
        statistic=statistic
    )
    precovid_lifespan = precovid_lifespan['2016-01-01':'2019-12-31']

    main_lifespan = process_data(
        posts_file,
        comments_file,
        aggregation=aggregation,
        units=units,
        statistic=statistic
    )

    combined_lifespan = pd.concat([precovid_lifespan, main_lifespan])

    agg_to_freq = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'yearly': 'Y'}
    freq_alias = agg_to_freq.get(aggregation, 'M')

    full_index = pd.date_range(
        start=combined_lifespan.index.min(),
        end=combined_lifespan.index.max(),
        freq=freq_alias
    )
    combined_lifespan = combined_lifespan.reindex(full_index)

    combined_lifespan = combined_lifespan.interpolate()

    if statistic in ['mean', 'median']:
        col_name = f'Post Lifespan ({statistic.capitalize()} in {units})'
    else:
        col_name = 'Number of Posts'

    metrics_table = pd.DataFrame(
        {col_name: combined_lifespan.values},
        index=combined_lifespan.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_response_times_data(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    time_unit='W',
    time_diff_unit='minutes'
):
    """
    Calculates the average response times for both pre-COVID and main datasets,
    then combines these series into a single-row DataFrame.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file.
    comments_file : str
        Path to the main comments Parquet file.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        Name of the city (unused in this version; kept for consistency).
    time_unit : str, optional
        Resampling frequency (default 'W' for weekly).
        Other options: 'D' (daily), 'M' (monthly), etc.
    time_diff_unit : str, optional
        Unit for response time. One of: 'seconds', 'minutes', 'hours', 'days'.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with columns as date strings (DD/MM/YYYY)
        and the row labeled as 'Average Response Time (...)'.
    """

    def process_data(one_posts_file, one_comments_file):
        """
        Reads a single pair of posts/comments files, computes the average response time,
        and returns it as a pandas Series indexed by the post creation time (resampled).
        """
        print(f"Reading posts data from {one_posts_file}...")
        posts = pd.read_parquet(one_posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")

        print(f"Reading comments data from {one_comments_file}...")
        comments = pd.read_parquet(one_comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")

        posts.rename(columns={'created_utc': 'created_utc_post'}, inplace=True)
        comments.rename(columns={'created_utc': 'created_utc_comment'}, inplace=True)

        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc_comment'], unit='s')

        posts['parent_id'] = 't3_' + posts['id']

        merged = comments.merge(
            posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
            on='parent_id',
            how='inner'
        )

        merged_sorted = merged.sort_values(['parent_id', 'created_datetime_comment'])
        first_comments = merged_sorted.groupby('parent_id').first().reset_index()

        first_comments['response_time_seconds'] = (
            first_comments['created_datetime_comment'] - first_comments['created_datetime_post']
        ).dt.total_seconds()

        if time_diff_unit == 'seconds':
            first_comments['response_time'] = first_comments['response_time_seconds']
        elif time_diff_unit == 'minutes':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 60
        elif time_diff_unit == 'hours':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 3600
        elif time_diff_unit == 'days':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 86400
        else:
            raise ValueError("Unsupported time_diff_unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")

        first_comments.set_index('created_datetime_post', inplace=True)
        avg_response = first_comments['response_time'].resample(time_unit).mean()

        return avg_response

    print("Processing pre-COVID data...")
    precovid_series = process_data(precovid_posts_file, precovid_comments_file)
    precovid_series = precovid_series['2016-01-01':'2019-12-31']

    print("Processing main data...")
    main_series = process_data(posts_file, comments_file)

    combined_series = pd.concat([precovid_series, main_series])

    full_index = pd.date_range(
        start=combined_series.index.min(),
        end=combined_series.index.max(),
        freq=time_unit
    )
    combined_series = combined_series.reindex(full_index)

    combined_series = combined_series.interpolate()

    if time_diff_unit == 'seconds':
        row_label = 'Average Response Time (Seconds)'
    elif time_diff_unit == 'minutes':
        row_label = 'Average Response Time (Minutes)'
    elif time_diff_unit == 'hours':
        row_label = 'Average Response Time (Hours)'
    elif time_diff_unit == 'days':
        row_label = 'Average Response Time (Days)'
    else:
        row_label = 'Average Response Time'

    metrics_df = pd.DataFrame(
        {row_label: combined_series.values},
        index=combined_series.index.strftime('%d/%m/%Y')
    ).transpose()

    print("Response time data for pre-COVID and main datasets combined. Returning DataFrame...")
    return metrics_df


def get_response_times_data_with_cutoff(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    time_unit='W',
    time_diff_unit='minutes'
):
    """
    Calculates the average response times for both pre-COVID and main datasets, 
    excluding any responses that occur more than 24 hours after a post is made.
    The resulting series are combined into a single-row DataFrame.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file.
    comments_file : str
        Path to the main comments Parquet file.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        Path to the pre-COVID comments Parquet file.
    city : str, optional
        Name of the city (unused in this version; kept for consistency).
    time_unit : str, optional
        Resampling frequency (default 'W' for weekly). Other options: 'D' (daily), 'M' (monthly), etc.
    time_diff_unit : str, optional
        Unit for response time. One of: 'seconds', 'minutes', 'hours', 'days'.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with columns as date strings (DD/MM/YYYY)
        and the row labeled as 'Average Response Time (...)'.
    """
    import pandas as pd

    def process_data(one_posts_file, one_comments_file):
        print(f"Reading posts data from {one_posts_file}...")
        posts = pd.read_parquet(one_posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")
        
        print(f"Reading comments data from {one_comments_file}...")
        comments = pd.read_parquet(one_comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")
        
        posts.rename(columns={'created_utc': 'created_utc_post'}, inplace=True)
        comments.rename(columns={'created_utc': 'created_utc_comment'}, inplace=True)
        
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc_comment'], unit='s')
        
        posts['parent_id'] = 't3_' + posts['id']
        
        merged = comments.merge(
            posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
            on='parent_id',
            how='inner'
        )
        
        merged_sorted = merged.sort_values(['parent_id', 'created_datetime_comment'])
        first_comments = merged_sorted.groupby('parent_id').first().reset_index()
        
        first_comments['response_time_seconds'] = (
            first_comments['created_datetime_comment'] - first_comments['created_datetime_post']
        ).dt.total_seconds()
        
        first_comments = first_comments[first_comments['response_time_seconds'] <= 86400]
        
        if time_diff_unit == 'seconds':
            first_comments['response_time'] = first_comments['response_time_seconds']
        elif time_diff_unit == 'minutes':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 60
        elif time_diff_unit == 'hours':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 3600
        elif time_diff_unit == 'days':
            first_comments['response_time'] = first_comments['response_time_seconds'] / 86400
        else:
            raise ValueError("Unsupported time_diff_unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")
        
        first_comments.set_index('created_datetime_post', inplace=True)
        avg_response = first_comments['response_time'].resample(time_unit).mean()
        
        return avg_response

    print("Processing pre-COVID data...")
    precovid_series = process_data(precovid_posts_file, precovid_comments_file)
    precovid_series = precovid_series['2016-01-01':'2019-12-31']
    
    print("Processing main data...")
    main_series = process_data(posts_file, comments_file)
    
    combined_series = pd.concat([precovid_series, main_series])
    
    full_index = pd.date_range(
        start=combined_series.index.min(),
        end=combined_series.index.max(),
        freq=time_unit
    )
    combined_series = combined_series.reindex(full_index)
    
    combined_series = combined_series.interpolate()
    
    if time_diff_unit == 'seconds':
        row_label = 'Average Response Time with Cutoff (Seconds)'
    elif time_diff_unit == 'minutes':
        row_label = 'Average Response Time with Cutoff (Minutes)'
    elif time_diff_unit == 'hours':
        row_label = 'Average Response Time with Cutoff (Hours)'
    elif time_diff_unit == 'days':
        row_label = 'Average Response Time with Cutoff (Days)'
    else:
        row_label = 'Average Response Time'
    
    metrics_df = pd.DataFrame(
        {row_label: combined_series.values},
        index=combined_series.index.strftime('%d/%m/%Y')
    ).transpose()
    
    print("Response time data (with 24-hour cutoff) for pre-COVID and main datasets combined. Returning DataFrame...")
    return metrics_df


def get_post_sentiment_metrics(
    posts_file,
    comments_file,  
    precovid_posts_file,
    precovid_comments_file, 
    city='',
    resample_unit='W',
    text_column='selftext'
):
    """
    Calculates and returns the average sentiment score metrics for posts, computed using VADER,
    for both pre-COVID and main datasets over time. Any periods with missing sentiment values are filled with 0.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing at least 'id', 'created_utc', and the text column.
    comments_file : str
        (Not used in this version, kept for consistency.)
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file.
    precovid_comments_file : str
        (Not used in this version, kept for consistency.)
    city : str, optional
        City name (not used in the computation; reserved for labeling if needed).
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).
    text_column : str, optional
        The column in the posts file that contains the post text (default 'selftext').

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row labeled 'Average Sentiment' and columns corresponding to
        resampled time periods (formatted as DD/MM/YYYY), containing the average sentiment score.
    """

    # Ensure the VADER lexicon is available.
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    def process_posts(posts_path):
        # Read only the necessary columns from the Parquet file.
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc', text_column])
        
        # Clean and filter out posts with missing or empty text.
        posts[text_column] = posts[text_column].fillna('').astype(str)
        posts = posts[posts[text_column].str.strip() != '']
        
        # Convert Unix timestamp to datetime and set as index.
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
        posts.set_index('created_datetime', inplace=True)
        
        # Compute sentiment for each post using VADER.
        posts['sentiment'] = posts[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # Resample the sentiment scores, compute the average, and fill any missing periods with 0.
        avg_sentiment = posts['sentiment'].resample(resample_unit).mean().fillna(0)
        return avg_sentiment

    # Process pre-COVID posts and restrict to the desired date range.
    precovid_sentiment = process_posts(precovid_posts_file)
    precovid_sentiment = precovid_sentiment['2016-01-01':'2019-12-31']

    # Process main posts.
    main_sentiment = process_posts(posts_file)

    # Combine the two time series.
    combined_sentiment = pd.concat([precovid_sentiment, main_sentiment])

    # Create a full date range index based on the combined data and reindex,
    # filling any missing periods with 0.
    full_index = pd.date_range(
        start=combined_sentiment.index.min(),
        end=combined_sentiment.index.max(),
        freq=resample_unit
    )
    combined_sentiment = combined_sentiment.reindex(full_index).fillna(0)

    # Construct the metrics table with date strings as column labels.
    metrics_table = pd.DataFrame(
        {'Average Sentiment': combined_sentiment.values},
        index=combined_sentiment.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_positive_sentiment_metrics(
    posts_file,
    precovid_posts_file,
    city='',
    text_column='selftext',
    resample_unit='W',
    threshold=0.05
):
    """
    Calculates and returns a metrics table of the number of posts with sentiment above a given threshold
    (e.g., 0.05) over time using VADER, for both pre‑COVID and main datasets. Pre‑COVID metrics are computed
    for the period 2016‑01‑01 to 2019‑12‑31. Any periods with missing sentiment values are filled with 0.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing at least 'id', 'created_utc', and the text column.
    precovid_posts_file : str
        Path to the pre‑COVID posts Parquet file.
    city : str, optional
        City name (or context) used for labeling (not directly used in calculations).
    text_column : str, optional
        The column in the posts file that contains the post text (default 'selftext').
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).
    threshold : float, optional
        Sentiment threshold above which a post is counted as "positive" (default 0.05).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row labeled 'Positive Sentiment Count'
        and columns corresponding to resampled time periods (formatted as DD/MM/YYYY),
        containing the positive sentiment post counts.
    """

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
        
    sid = SentimentIntensityAnalyzer()

    def process_posts(posts_path):
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc', text_column])
        
        # Clean and filter out posts with missing or empty text.
        posts[text_column] = posts[text_column].fillna('').astype(str)
        posts = posts[posts[text_column].str.strip() != '']
        
        # Convert Unix timestamp to datetime and set as index.
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
        posts.set_index('created_datetime', inplace=True)
        
        # Compute sentiment scores using VADER.
        posts['sentiment'] = posts[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        # Identify positive posts based on the threshold.
        posts['is_positive'] = (posts['sentiment'] > threshold).astype(int)
        
        # Resample the positive post counts over the specified time unit and fill missing periods with 0.
        positive_count = posts['is_positive'].resample(resample_unit).sum().fillna(0)
        return positive_count

    # Process pre‑COVID posts and restrict to the desired date range.
    precovid_positive = process_posts(precovid_posts_file)
    precovid_positive = precovid_positive['2016-01-01':'2019-12-31']

    # Process main posts.
    main_positive = process_posts(posts_file)

    # Combine the two time series.
    combined_positive = pd.concat([precovid_positive, main_positive])

    # Create a full date range index based on the combined data and reindex,
    # filling any missing periods with 0.
    full_index = pd.date_range(
        start=combined_positive.index.min(),
        end=combined_positive.index.max(),
        freq=resample_unit
    )
    combined_positive = combined_positive.reindex(full_index).fillna(0)

    # Construct the metrics table with date strings as column labels.
    metrics_table = pd.DataFrame(
        {'Positive Sentiment Count': combined_positive.values},
        index=combined_positive.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def get_negative_sentiment_metrics(
    posts_file,
    precovid_posts_file,
    city='',
    text_column='selftext',
    resample_unit='W',
    threshold=-0.05
):
    """
    Calculates and returns a metrics table of the number of posts with sentiment below a given threshold
    (e.g., -0.05) over time using VADER, for both pre‑COVID and main datasets. Pre‑COVID metrics are computed
    for the period 2016‑01‑01 to 2019‑12‑31. Any periods with missing sentiment values are filled with 0.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing at least 'id', 'created_utc', and the text column.
    precovid_posts_file : str
        Path to the pre‑COVID posts Parquet file.
    city : str, optional
        City name (or context) used for labeling (not directly used in calculations).
    text_column : str, optional
        The column in the posts file that contains the post text (default 'selftext').
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).
    threshold : float, optional
        Sentiment threshold below which a post is counted as "negative" (default -0.05).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row labeled 'Negative Sentiment Count'
        and columns corresponding to resampled time periods (formatted as DD/MM/YYYY).
    """

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
        
    sid = SentimentIntensityAnalyzer()

    def process_posts(posts_path):
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc', text_column])
        
        posts[text_column] = posts[text_column].fillna('').astype(str)
        posts = posts[posts[text_column].str.strip() != '']
    
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
        posts.set_index('created_datetime', inplace=True)
        
        posts['sentiment'] = posts[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        
        posts['is_negative'] = (posts['sentiment'] < threshold).astype(int)
        
        negative_post_count = posts['is_negative'].resample(resample_unit).sum().fillna(0)
        return negative_post_count

    precovid_negative = process_posts(precovid_posts_file)
    precovid_negative = precovid_negative['2016-01-01':'2019-12-31']

    main_negative = process_posts(posts_file)

    combined_negative = pd.concat([precovid_negative, main_negative])

    full_index = pd.date_range(
        start=combined_negative.index.min(),
        end=combined_negative.index.max(),
        freq=resample_unit
    )
    combined_negative = combined_negative.reindex(full_index).fillna(0)

    metrics_table = pd.DataFrame(
        {'Negative Sentiment Count': combined_negative.values},
        index=combined_negative.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def save_timeseries_metrics_for_cities(city_dict):
    
    output_dir = "../metrics"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        prophet_train_submissions_path = f"../prophet_train_parquet/{city_lower}_submissions.parquet"
        prophet_train_comments_path = f"../prophet_train_parquet/{city_lower}_comments.parquet"

        post_count_metrics = get_raw_post_count_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'
        )

        comment_count_metrics = get_raw_comment_count_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'
        )

        comment_percentage_metrics = get_comment_percentage_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'
        )

        lifespan_metrics = get_post_lifespan_timeseries(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            aggregation='weekly',
            units='hours',
            statistic='mean'
        )

        response_metrics = get_response_times_data(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            time_unit='W',
            time_diff_unit='minutes'
        )

        cutoff_response_metrics = get_response_times_data_with_cutoff(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            time_unit='W',
            time_diff_unit='minutes'
        )

        post_sentiment_metrics = get_post_sentiment_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W',
            text_column='selftext'
        )

        positive_sentiment_metrics = get_positive_sentiment_metrics(
            submissions_path,
            prophet_train_submissions_path,
            city=city_lower,
            text_column='selftext',
            resample_unit='W',
            threshold=0.05
        )

        negative_sentiment_metrics = get_negative_sentiment_metrics(
            submissions_path,
            prophet_train_submissions_path,
            city=city_lower,
            text_column='selftext',
            resample_unit='W',
            threshold=-0.05
        )

        combined_metrics = pd.concat([
            post_count_metrics, 
            comment_count_metrics, 
            comment_percentage_metrics, 
            lifespan_metrics, 
            response_metrics, 
            cutoff_response_metrics,
            post_sentiment_metrics,
            positive_sentiment_metrics,
            negative_sentiment_metrics
        ], axis=0)

        output_path = os.path.join(output_dir, f"{city_lower}_metrics.parquet")
        combined_metrics.to_parquet(output_path)


def save_timeseries_metrics_for_cities_2(city_dict, input_dir=None, output_dir="../metrics_test"):
    """
    For each city in `city_dict`, generate various time-series metrics and then either
    update an existing parquet file (if found in input_dir) or write a new file in output_dir.

    If a file exists (from input_dir) for a given city, then only new metric data (i.e., rows whose
    'metric' value is not already present) is appended. Otherwise, if every metric is already in the file,
    nothing is written.
    """
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        prophet_train_submissions_path = f"../prophet_train_parquet/{city_lower}_submissions.parquet"
        prophet_train_comments_path = f"../prophet_train_parquet/{city_lower}_comments.parquet"

        # post_count_metrics = get_raw_post_count_metrics(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     resample_unit='W'
        # ).copy()
        # post_count_metrics['metric'] = 'post_count'

        # comment_count_metrics = get_raw_comment_count_metrics(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     resample_unit='W'
        # ).copy()
        # comment_count_metrics['metric'] = 'comment_count'

        # comment_percentage_metrics = get_comment_percentage_metrics(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     resample_unit='W'
        # ).copy()
        # comment_percentage_metrics['metric'] = 'comment_percentage'

        # lifespan_metrics = get_post_lifespan_timeseries(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     aggregation='weekly',
        #     units='hours',
        #     statistic='mean'
        # ).copy()
        # lifespan_metrics['metric'] = 'lifespan'

        # response_metrics = get_response_times_data(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     time_unit='W',
        #     time_diff_unit='minutes'
        # ).copy()
        # response_metrics['metric'] = 'response'

        # cutoff_response_metrics = get_response_times_data_with_cutoff(
        #     submissions_path,
        #     comments_path,
        #     prophet_train_submissions_path,
        #     prophet_train_comments_path,
        #     city=city_lower,
        #     time_unit='W',
        #     time_diff_unit='minutes'
        # ).copy()
        # cutoff_response_metrics['metric'] = 'cutoff_response'

        # combined_metrics = pd.concat([
        #     post_count_metrics, 
        #     comment_count_metrics, 
        #     comment_percentage_metrics, 
        #     lifespan_metrics, 
        #     response_metrics, 
        #     cutoff_response_metrics,
        # ], axis=0)

        post_sentiment_metrics = get_post_sentiment_metrics(
            submissions_path,
            comments_path,  
            prophet_train_submissions_path,
            prophet_train_submissions_path, 
            city=city_lower,
            resample_unit='W',
            text_column='selftext'
        )
        post_sentiment_metrics = post_sentiment_metrics.copy()
        post_sentiment_metrics['metric'] = 'post_sentiment'

        combined_metrics = pd.concat([
            post_sentiment_metrics,
        ], axis=0)

        file_name = f"{city_lower}_metrics.parquet"

        if input_dir is not None:
            target_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            if os.path.exists(target_path):
                existing_metrics = pd.read_parquet(target_path)

                # If the file already has a 'metric' column, we can safely check for duplicates.
                if 'metric' in existing_metrics.columns:
                    existing_metric_names = set(existing_metrics['metric'].unique())
                    # Identify rows in the new data that have a metric not already in the file.
                    new_rows = combined_metrics[~combined_metrics['metric'].isin(existing_metric_names)]

                    if not new_rows.empty:
                        # Merge old and new metrics.
                        merged_metrics = pd.concat([existing_metrics, new_rows], axis=0)
                        merged_metrics.to_parquet(output_path)
                        print(f"Updated {output_path} with new metrics: {new_rows['metric'].unique().tolist()}")
                    else:
                        print(f"No new metrics for {city_lower} in {target_path}.")
                    # Finished updating this city; move on to the next.
                    continue
                else:
                    # If the file exists but does not have a 'metric' column, assume its structure is unexpected.
                    # Overwrite it with the combined metrics.
                    combined_metrics.to_parquet(output_path)
                    print(f"Overwrote {output_path} (no 'metric' column found) with combined metrics.")
                    continue

        # If no matching file was found in input_dir, write a new file to the output directory.
        output_path = os.path.join(output_dir, file_name)
        combined_metrics.to_parquet(output_path)
        print(f"Wrote new metrics file {output_path}.")


def aggregate_metrics_by_cluster(metrics_path, clusters_path, output_path, metric, aggregation="mean"):
    clusters_df = pd.read_parquet(clusters_path)
    clusters = clusters_df.set_index("City")["Cluster"]

    cluster_aggregates = {}

    print("Processing metrics for each city...")
    for city_file in os.listdir(metrics_path):
        if city_file.endswith(".parquet"):
            city_name = os.path.splitext(city_file)[0].replace("_metrics", "")

            if city_name not in clusters:
                print(f"Warning: {city_name} is not in the cluster assignments. Skipping.")
                continue

            cluster_id = clusters[city_name]

            city_metrics_path = os.path.join(metrics_path, city_file)
            city_df = pd.read_parquet(city_metrics_path)
            
            city_df.columns = pd.to_datetime(city_df.columns, format="%d/%m/%Y")
            city_df = city_df.T 

            if metric not in city_df.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in file columns. Available metrics: {list(city_df.columns)}"
                )
            
            city_df = city_df[[metric]]

            if cluster_id not in cluster_aggregates:
                cluster_aggregates[cluster_id] = []
            cluster_aggregates[cluster_id].append(city_df)

    aggregated_results = {}
    for cluster_id, cluster_data in cluster_aggregates.items():
        print(f"Aggregating metrics for Cluster {cluster_id}...")
        combined_df = pd.concat(cluster_data, axis=1)
        if aggregation == "mean":
            aggregated_df = combined_df.mean(axis=1)
        elif aggregation == "median":
            aggregated_df = combined_df.median(axis=1)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        aggregated_results[f"Cluster {cluster_id}"] = aggregated_df

    final_df = pd.DataFrame(aggregated_results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_df.to_parquet(output_path)

    print(f"Aggregated metrics saved to {output_path}")


def aggregate_metrics_by_city(metrics_path, output_path, metric, aggregation="mean"):
    """
    Reads all parquet files in metrics_path, where each file contains metrics for one city.
    For each city, it extracts the specified metric (after converting the columns to datetime
    and transposing the DataFrame so that dates become the index). Then, it aggregates the 
    metric across all cities using the specified method (mean or median) and saves the result 
    as a parquet file.

    Parameters:
      metrics_path (str): Directory containing city metric parquet files.
      output_path (str): Path to save the aggregated parquet file.
      metric (str): The metric to aggregate.
      aggregation (str): Aggregation method ("mean" or "median"). Default is "mean".

    Returns:
      None
    """
    city_data = []
    
    print("Processing metrics for each city...")
    for city_file in os.listdir(metrics_path):
        if city_file.endswith(".parquet"):
            city_name = os.path.splitext(city_file)[0].replace("_metrics", "")
            city_metrics_path = os.path.join(metrics_path, city_file)
            city_df = pd.read_parquet(city_metrics_path)    
            city_df.columns = pd.to_datetime(city_df.columns, format="%d/%m/%Y")
            city_df = city_df.T
            
            if metric not in city_df.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in file '{city_file}'. Available metrics: {list(city_df.columns)}"
                )
            
            metric_df = city_df[[metric]].copy()
            metric_df.rename(columns={metric: city_name}, inplace=True)
            city_data.append(metric_df)
    
    if not city_data:
        print("No city data found.")
        return

    print("Averaging metrics over all cities...")
    # Concatenate all city dataframes along columns (aligning on dates)
    combined_df = pd.concat(city_data, axis=1)
    
    # Aggregate over cities using the specified method.
    if aggregation == "mean":
        aggregated_series = combined_df.mean(axis=1)
    elif aggregation == "median":
        aggregated_series = combined_df.median(axis=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    # Convert the resulting series into a DataFrame.
    final_df = pd.DataFrame({metric: aggregated_series})
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_df.to_parquet(output_path)
    
    print(f"Aggregated metrics saved to {output_path}")


def print_parquet(file_path):
    df = pd.read_parquet(file_path)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)

    print(df)
    
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.expand_frame_repr")


def fill_nulls_with_zero(file_path, output_path):
    """
    Loads a parquet file, fills all null values with 0, and saves the cleaned DataFrame.

    Parameters:
    - file_path: str, path to the input parquet file.
    - output_path: str, path where the cleaned parquet file will be saved.
    """
    df = pd.read_parquet(file_path)
    df_filled = df.fillna(0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filled.to_parquet(output_path)
    return df_filled


def normalise_timeseries(input_path, output_path):
    """
    Normalise the disaster period (2020–2022) data in a parquet file of time series.
    
    The input parquet file is expected to have:
      - Each row as a distinct metric.
      - Each column as a date (weekly, represented as a string or datetime).
    
    The normalisation process is as follows:
      1. Convert the columns to a DatetimeIndex.
      2. Optionally resample the data to a given frequency (using the mean).
      3. Define baseline (pre-disaster) period: 2016-01-01 to 2019-12-31.
      4. Define disaster period: 2020-01-01 to 2022-12-31.
      5. For each metric (row):
         a. Compute the baseline mean (μ) and population standard deviation (σ) 
            using data from the baseline period.
         b. For each date in the disaster period, compute the normalised value:
                (value - μ) / σ.
         c. (If σ is zero for a metric, normalised disaster values are set to NaN.)
      6. The baseline period data are left unchanged.
      7. Save the resulting DataFrame (with metrics as rows and dates as columns) 
         to the output parquet file.
    
    Parameters:
      input_path (str): File path to the input parquet file.
      output_path (str): File path to save the output parquet file.
      resample_unit (str, optional): A pandas offset alias (e.g., 'W' for weekly, 
                                     'M' for monthly). If provided, the data are resampled 
                                     using the mean.
    """
    df = pd.read_parquet(input_path)
    df = df.T

    df.index = pd.to_datetime(df.index, format="%d/%m/%Y")
    
    baseline_start = pd.to_datetime("2016-01-01")
    baseline_end   = pd.to_datetime("2016-12-31")
    disaster_start = pd.to_datetime("2017-01-01")
    disaster_end   = pd.to_datetime("2021-12-31")
    
    baseline_mask = (df.index >= baseline_start) & (df.index <= baseline_end)
    disaster_mask = (df.index >= disaster_start) & (df.index <= disaster_end)

    normalised_df = df.copy()
    
    for metric in df.columns:
        baseline_values = df.loc[baseline_mask, metric]
        disaster_values = df.loc[disaster_mask, metric]
        
        mu = baseline_values.mean()
        sigma = baseline_values.std(ddof=0)

        print(mu, sigma)
        
        if sigma == 0:
            normalised_values = np.full(disaster_values.shape, np.nan)
        else:
            normalised_values = (disaster_values - mu) / sigma
        
        normalised_df.loc[disaster_mask, metric] = normalised_values
    
    normalised_df = normalised_df.T
    normalised_df.columns = normalised_df.columns.strftime("%d/%m/%Y")
    
    normalised_df.to_parquet(output_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalised_df.to_parquet(output_path)


def smooth_timeseries(file_path, output_path, resample_unit='W'):
    df = pd.read_parquet(file_path)
    
    try:
        df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
    except Exception as e:
        raise ValueError(f"Error converting column headers to datetime in {file_path}: {e}")
    
    df = df.T
    df.index.name = "datetime"
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print("Data range:", df.index.min(), "to", df.index.max())
    
    df = df.loc["2017-01-01":"2021-12-31"]
    
    df_resampled = df.resample(resample_unit).mean().interpolate()

    smoothed_series = {}
    x = range(len(df_resampled))

    for metric in df_resampled.columns:
        y = df_resampled[metric].values
        # smoothed = sm.nonparametric.lowess(y, x, frac=0.2) # liwc
        smoothed = sm.nonparametric.lowess(y, x, frac=0.1) # less aggressive test
        smoothed_y = smoothed[:, 1]
        print(f"Metric: {metric}, length of smoothed_y: {len(smoothed_y)}")
        smoothed_series[metric] = smoothed_y

    L = len(next(iter(smoothed_series.values())))
    smoothed_df = pd.DataFrame(smoothed_series, index=df_resampled.index[:L]).T
    smoothed_df.columns = smoothed_df.columns.strftime("%d/%m/%Y")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    smoothed_df.to_parquet(output_path)
    
    return smoothed_df

if __name__ == "__main__":

    cities = [
        "albuquerque", 
        "anchorage", 
        "arlington", 
        "atlanta", 
        "aurora", 
        "austin",
        "bakersfield", 
        "baltimore", 
        "batonrouge", 
        "boise", 
        "boston", 
        "buffalo",
        "charlotte", 
        "chicago", 
        "cincinnati", 
        "cleveland", 
        "coloradosprings",
        "columbus", 
        "corpuschristi", 
        "dallas", 
        "denver", 
        "detroit", 
        "elpaso",
        "fortwayne", 
        "fortworth", 
        "fremont", 
        "fresno", 
        # "frisco", 
        "glendale",
        "honolulu", 
        "houston", 
        # "huntsville",
        "indianapolis", 
        "irvine", 
        "jacksonville", 
        "jerseycity", 
        "kansascity",
        "laredo", 
        "lasvegas", 
        "lexington", 
        "lincoln", 
        "longbeach", 
        "losangeles",
        "louisville", 
        "lubbock", 
        "madison", 
        "memphis", 
        "miami", 
        "milwaukee",
        "minneapolis", 
        "nashville", 
        "newark", 
        "neworleans", 
        "norfolk", 
        "newyorkcity",
        "oakland", 
        "oklahomacity", 
        "omaha", 
        "orlando", 
        "philadelphia",
        "pittsburgh", 
        "plano", 
        "portland",
        "reno", 
        "richmond", 
        "riverside", 
        "richmond", 
        "sacramento",
        "saintpaul",
        "sanantonio", 
        "sandiego", 
        "sanfrancisco", 
        "sanjose", 
        "santaclarita",
        "scottsdale", 
        "seattle", 
        "spokane", 
        "stlouis", 
        "stockton",
        "stpetersburg", 
        "tampa", 
        "toledo", 
        "tucson", 
        "tulsa", 
        "virginiabeach",
        "washingtondc", 
        "wichita", 
        "winstonsalem"
    ]

    metrics_dict = {
        "Percentage of Posts with Comments": "comments_percentage",
        "Raw Number of Posts": "submissions",
        "Raw Number of Comments": "comments",
        "Post Lifespan (Mean in hours)": "lifespan",
        "Average Response Time with Cutoff (Minutes)": "response",
        "Average Sentiment": "average_sentiment",
        "Positive Sentiment Count": "positive_sentiment",
        "Negative Sentiment Count": "negative_sentiment",
    }


    # for city in cities:
    #     normalise_timeseries(f"../metrics_filled/{city}_metrics.parquet", f"../metrics_filled_normalised2/{city}_metrics.parquet")

    # for city in cities:
    #     smooth_timeseries(f"../metrics_filled_normalised2/{city}_metrics.parquet", f"../metrics_filled_normalised_smoothed4_01/{city}_metrics.parquet", resample_unit="W")

    # print_parquet("../metric_clusters/comments_percentage_clusters.parquet")

    # aggregate_metrics_by_cluster(
    #     metrics_path="../metrics",
    #     clusters_path="../metric_clusters/submissions_clusters.parquet",
    #     output_path="../cluster_aggregated_metrics/aggregated_submissions.parquet",
    #     metric="Raw Number of Posts",
    #     aggregation="mean"
    # )

    # for metric_key, metric_name in metrics_dict.items():
    #     aggregate_metrics_by_cluster(
    #         metrics_path="../metrics",
    #         clusters_path=f"../metric_fns_clusters/{metric_name}_clusters.parquet",
    #         output_path=f"../cluster_fns_aggregated_metrics/aggregated_{metric_name}.parquet",
    #         metric=metric_key,
    #         aggregation="mean"
    #     )

    # for metric_key, metric_name in metrics_dict.items():
    #     aggregate_metrics_by_cluster(
    #         metrics_path="../metrics_filled_normalised_smoothed2",
    #         clusters_path=f"../metric_fns_clusters2/{metric_name}_clusters.parquet",
    #         output_path=f"../cluster_fns_aggregated_fns_metrics2/aggregated_{metric_name}.parquet",
    #         metric=metric_key,
    #         aggregation="mean"
    #     )

    for metric_key, metric_name in metrics_dict.items():
        aggregate_metrics_by_cluster(
            metrics_path="../metrics_filled_normalised_smoothed4_01",
            clusters_path=f"../metric_fns_clusters4_01/{metric_name}_clusters.parquet",
            output_path=f"../cluster_fns_aggregated_fns_metrics4_01/aggregated_{metric_name}.parquet",
            metric=metric_key,
            aggregation="mean"
        )

    # for metric_key, metric_name in metrics_dict.items():
    #     aggregate_metrics_by_cluster(
    #         metrics_path="../metrics",
    #         clusters_path=f"../metric_clusters/{metric_name}_clusters.parquet",
    #         output_path=f"../cluster_aggregated_metrics/aggregated_{metric_name}.parquet",
    #         metric=metric_key,
    #         aggregation="mean"
    #     )

    # for city in cities:
    #     input_path = f"../metrics/{city}_metrics.parquet"
    #     output_path = f"../metrics_filled/{city}_metrics.parquet"
    #     fill_nulls_with_zero(input_path, output_path)
    #     print(f"Processed {city}: filled nulls and saved to {output_path}")

