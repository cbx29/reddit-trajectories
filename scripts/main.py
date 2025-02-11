#!/usr/bin/env python3

import sys
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.stats as stats
from scipy.stats import spearmanr
import subprocess
from pathlib import Path
from typing import Dict, Set, Optional
import seaborn as sns
import plotly.express as px
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dtaidistance import dtw
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from joblib import Parallel, delayed

def extract_data_between_dates(input_file_path, output_file_path, start_date_str, end_date_str):
    """
    Extracts data from an NDJSON file within a specified date range and writes them to a JSON array.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)

    filtered_data = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)
                created_utc = entry.get('created_utc')
                if created_utc is not None:
                    if isinstance(created_utc, str):
                        created_utc = int(float(created_utc))
                    created_date = datetime.utcfromtimestamp(created_utc)
                    if start_date <= created_date <= end_date:
                        filtered_data.append(entry)

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {line_num} in {input_file_path}: {e}")
                continue

    if not filtered_data:
        print(f"No data found in date range for file: {input_file_path}")
        return False

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


def num_lines(file):
    with open(file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return len(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {file}: {e}")
            return 0


def filter_submissions(input_file, output_file):
    """
    Filters submissions to keep only specified fields.
    """
    fields_to_keep = {"author", "author_created_utc", "created_utc", "num_comments", "selftext", "title", "id"}

    with open(input_file, "r", encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {input_file}: {e}")
            return False

    if not data:
        print(f"No data to filter in file: {input_file}")
        return False

    filtered_data = [{k: v for k, v in entry.items() if k in fields_to_keep} for entry in data]

    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


def filter_comments(input_file, output_file):
    """
    Filters comments to keep only specified fields.
    """
    fields_to_keep = {"author", "body", "created_utc", "id", "parent_id", "subreddit", "ups", "downs", "score", "distinguished"}

    with open(input_file, "r", encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {input_file}: {e}")
            return False

    if not data:
        print(f"No data to filter in file: {input_file}")
        return False

    filtered_data = [{k: v for k, v in entry.items() if k in fields_to_keep} for entry in data]

    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


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


def json_file_to_parquet(json_file_path, output_file):
    """
    Converts a JSON file containing an array to a Parquet file.

    Parameters:
        json_file_path (str): Path to the JSON file.
        output_file (str): Path to save the Parquet file.

    Returns:
        None
    """
    df = pd.read_json(json_file_path)
    
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Parquet file saved to {output_file}")


def remove_bot_entries_parquet(input_file, output_file):
    bot_pattern = re.compile(r'(?:^|\b|_)(bot)(?:$|\b|_)', re.IGNORECASE)
    
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error reading Parquet file '{input_file}': {e}")
        return False

    if df.empty:
        print(f"No data to filter in file: '{input_file}'")
        return False

    is_bot_author = df['author'].str.contains(bot_pattern)
    
    bot_authors = df.loc[is_bot_author, 'author'].unique()

    df_filtered = df.loc[~is_bot_author].copy()

    try:
        df_filtered.to_parquet(output_file, index=False)
        print(f"Filtered data saved to '{output_file}'")
        return True
    except Exception as e:
        print(f"Error writing Parquet file '{output_file}': {e}")
        return False


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


def count_unique_authors(parquet_file):
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


def process_city_files(city_names):
    raw_dir = "../raw"
    output_dir = "../raw_json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for city in city_names:
        city_raw_dir = os.path.join(raw_dir, city)
        if not os.path.exists(city_raw_dir):
            print(f"Directory for city '{city}' not found in {raw_dir}. Skipping...")
            continue
        
        submissions_zst = os.path.join(city_raw_dir, f"{city}_submissions.zst")
        comments_zst = os.path.join(city_raw_dir, f"{city}_comments.zst")
        submissions_json = os.path.join(output_dir, f"{city}_submissions.json")
        comments_json = os.path.join(output_dir, f"{city}_comments.json")
        
        if os.path.exists(submissions_zst):
            print(f"Processing {submissions_zst}...")
            subprocess.run(
                f"zstd -d {submissions_zst} -o {submissions_json}", 
                shell=True, 
                check=True
            )
        else:
            print(f"File {submissions_zst} not found. Skipping...")
        
        if os.path.exists(comments_zst):
            print(f"Processing {comments_zst}...")
            subprocess.run(
                f"zstd -d {comments_zst} -o {comments_json}", 
                shell=True, 
                check=True
            )
        else:
            print(f"File {comments_zst} not found. Skipping...")

    print("Processing complete.")


def convert_jsons_to_parquets_efficient(json_dir, parquet_dir, chunk_size=100000):
    """
    Converts large JSON files in a directory to Parquet files in chunks for efficiency.

    Parameters:
        json_dir (str): Path to the directory containing JSON files.
        parquet_dir (str): Path to the directory where Parquet files will be saved.
        chunk_size (int): Number of rows per chunk to process. Default is 100,000.

    Returns:
        None
    """
    os.makedirs(parquet_dir, exist_ok=True)
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_file_path = os.path.join(json_dir, json_file)
            parquet_file_name = json_file.replace(".json", ".parquet")
            parquet_file_path = os.path.join(parquet_dir, parquet_file_name)
            
            try:
                print(f"Processing {json_file_path} into {parquet_file_path} in chunks...")
                
                chunks = pd.read_json(
                    json_file_path,
                    lines=True,
                    chunksize=chunk_size
                )
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        chunk.to_parquet(parquet_file_path, engine='pyarrow', index=False)
                    else:
                        chunk.to_parquet(parquet_file_path, engine='pyarrow', index=False, append=True)
                
                print(f"Finished converting {json_file_path} to {parquet_file_path}.")
            
            except Exception as e:
                print(f"Failed to process {json_file_path}: {e}")

    print("All JSON-to-Parquet conversions complete.")


def convert_json_to_array(input_dir, output_dir):
    """
    Converts line-delimited JSON files in a directory to JSON array format.

    Parameters:
        input_dir (str): Path to the directory containing the input JSON files.
        output_dir (str): Path to the directory where the converted JSON files will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in os.listdir(input_dir):
        if json_file.endswith(".json"):
            input_file_path = os.path.join(input_dir, json_file)
            output_file_path = os.path.join(output_dir, json_file)
            
            try:
                print(f"Converting {input_file_path} to JSON array format...")
                
                with open(input_file_path, "r") as infile:
                    json_objects = [json.loads(line) for line in infile]
                
                with open(output_file_path, "w") as outfile:
                    json.dump(json_objects, outfile, indent=2)
                
                print(f"Converted file saved to {output_file_path}")
            
            except Exception as e:
                print(f"Failed to process {input_file_path}: {e}")

    print("Conversion to JSON arrays complete.")


def filter_and_extract_json(input_file, output_file, type=None, start_date_str=None, end_date_str=None):
    """
    Filters and extracts data from a JSON/NDJSON file based on type-specific fields and optional date range.

    Parameters:
        input_file (str): Path to the input JSON/NDJSON file.
        output_file (str): Path to the output JSON file.
        type (str): Type of data ("comment" or "submission"). Determines which fields to keep.
        start_date_str (str): Start date (inclusive) in 'YYYY-MM-DD' format for filtering by date. Default is None.
        end_date_str (str): End date (inclusive) in 'YYYY-MM-DD' format for filtering by date. Default is None.

    Returns:
        bool: True if the filtering was successful, False otherwise.
    """
    fields_to_keep = {
        "submission": {"author", "author_created_utc", "created_utc", "num_comments", "selftext", "title", "id", "ups", "downs", "score"},
        "comment": {"author", "body", "created_utc", "id", "parent_id", "subreddit", "ups", "downs", "score", "distinguished"}
    }

    if type not in fields_to_keep:
        raise ValueError(f"Invalid type '{type}'. Must be 'comment' or 'submission'.")

    selected_fields = fields_to_keep[type]

    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    else:
        start_date = end_date = None

    filtered_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)
                if start_date and end_date:
                    created_utc = entry.get('created_utc')
                    if created_utc is not None:
                        if isinstance(created_utc, str):
                            created_utc = int(float(created_utc))
                        created_date = datetime.utcfromtimestamp(created_utc)
                        if not (start_date <= created_date <= end_date):
                            continue

                entry = {k: v for k, v in entry.items() if k in selected_fields}
                filtered_data.append(entry)

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {line_num} in {input_file}: {e}")
                continue

    if not filtered_data:
        print(f"No data found matching criteria in file: {input_file}")
        return False

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


def user_population_spearman(city_users, city_population):
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


def plot_post_count(
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


def plot_comment_count(
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


def plot_post_lifespan_timeseries(
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
    

def plot_response_times(posts_file, comments_file, city='', time_unit='W', time_diff_unit='minutes', save_plot=False, plot_path='average_response_times.png', display_plot=True):
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


def plot_response_times_with_cutoff(
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


def plot_comment_percentage_time_window(
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
    import pandas as pd
    import matplotlib.pyplot as plt

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


def plot_post_sentiment(posts_file, city='', text_column='selftext', time_unit='W', save_plot=False, plot_path='post_sentiment.png', display_plot=True):
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


def plot_negative_sentiment_count(
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


def plot_positive_sentiment_count(
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


def remove_automoderator_data(city_names, source_folder="../covid_data_parquet", target_folder="../covid_data_parquet_2"):
    """
    Processes .parquet files for a list of city names by removing entries where the author is "AutoModerator".

    Parameters:
        city_names (list): List of city names.
        source_folder (str): Path to the folder containing the source .parquet files.
        target_folder (str): Path to the folder to save the processed .parquet files.
    """
    os.makedirs(target_folder, exist_ok=True)

    for city in city_names:
        for file_type in ["comments", "submissions"]:
            file_name = f"{city}_{file_type}.parquet"
            source_path = os.path.join(source_folder, file_name)

            if os.path.exists(source_path):
                try:
                    df = pd.read_parquet(source_path)

                    filtered_df = df[df['author'] != "AutoModerator"]

                    target_path = os.path.join(target_folder, file_name)
                    filtered_df.to_parquet(target_path, index=False)

                    print(f"Processed and saved: {target_path}")
                except Exception as e:
                    print(f"Error processing {source_path}: {e}")
            else:
                print(f"File not found: {source_path}")


def generate_graphs_for_cities(city_dict):
    """
    Generates and saves graphs for each city contained in city_dict.

    Parameters:
    - city_dict (dict): Dictionary where each key is a city identifier (used in file paths and filenames)
                        and each value is the properly capitalized city name (for plot titles).

    Returns:
    - None: Graphs are saved as PNG files in the "graphs" folder.
    """
    output_dir = "../graphs"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        precovid_submissions_path = f"../precovid_data_parquet/{city_lower}_submissions.parquet"
        precovid_comments_path = f"../precovid_data_parquet/{city_lower}_comments.parquet"

        # # --- Plot post sentiment ---
        # plot_post_sentiment(submissions_path, city_name, text_column='selftext', time_unit='W', save_plot=True, plot_path=f'../graphs/{city_key}_post_sentiment.png', display_plot=False)
        # # sentiment_filename = os.path.join(output_dir, f"{city_key}_post_sentiment.png")
        # # plt.savefig(sentiment_filename)
        # plt.clf()

        # plot_negative_sentiment_count(
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

        # plot_positive_sentiment_count(
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

        # --- Plot comment percentage ---
        # plot_comment_percentage_time_window(
        #     submissions_path,
        #     comments_path,
        #     city=city_name,
        #     resample_unit='W',
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_comment_percentage_same_week.png',
        #     display_plot=False
        # )

        # plot_comment_percentage_time_window(
        #     submissions_path,
        #     comments_path,
        #     precovid_submissions_path,
        #     precovid_comments_path,
        #     city=city_name,
        #     resample_unit='W',
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_comment_percentage_same_week.png',
        #     display_plot=False
        # )

        # plot_post_count(
        #     submissions_path, 
        #     comments_path,
        #     precovid_submissions_path,
        #     precovid_comments_path, 
        #     city=city_name, 
        #     resample_unit='W', 
        #     save_plot=True, 
        #     plot_path=f'../graphs/{city_key}_post_count_timeseries.png', 
        #     display_plot=False
        # )

        # plt.clf()

        # plot_comment_count(
        #     submissions_path, 
        #     comments_path,
        #     precovid_submissions_path, 
        #     precovid_comments_path, 
        #     city=city_name, 
        #     resample_unit='W', 
        #     save_plot=True, 
        #     plot_path=f'../graphs/{city_key}_comment_count_timeseries.png', 
        #     display_plot=False
        # )

        plot_response_times_with_cutoff(
            submissions_path,
            comments_path,
            city=city_name,
            time_unit='W',
            time_diff_unit='hours',
            save_plot=True,
            plot_path=f'../graphs/{city_key}_average_response_times_cutoff.png',
            display_plot=False
        )

        plt.clf()

        # # --- Plot response times ---
        # plot_response_times(submissions_path, comments_path, city_name, time_unit='W', time_diff_unit='hours', save_plot=True, plot_path=f'../graphs/{city_key}_average_response_times.png', display_plot=False)
        # plt.clf()

        # # --- Plot post lifespan timeseries ---
        # plot_post_lifespan_timeseries(
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
        # plt.clf()

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

        print(f"Graphs for {city_name} saved in the '{output_dir}' folder.")


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
    import pandas as pd
    import numpy as np

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


def load_liwc_dictionary(dictionary_path):
    liwc_dict = {}
    with open(dictionary_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            try:
                word, categories = line.split('\t', 1)
                liwc_dict[word.lower()] = categories.split('\t')
            except ValueError:
                print(f"Skipping line: {line}")
    return liwc_dict


def load_category_mapping(mapping_file_path):
    category_map = {}
    with open(mapping_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            try:
                code, name = line.split('\t')
                category_map[code] = name
            except ValueError:
                print(f"Skipping malformed line: {line}")
    return category_map


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def analyze_emotion(text, liwc_dict, category_map, emotion_word):
    tokens = tokenize(text)
    total_words = len(tokens)
    
    if total_words == 0:
        return 0

    count = 0
    for token in tokens:
        if token in liwc_dict:
            for code in liwc_dict[token]:
                if category_map.get(code, "").lower() == emotion_word.lower():
                    count += 1

    percentage = (count / total_words) * 100
    return percentage


def get_liwc_percentage_metrics(
    posts_file,
    precovid_posts_file,
    liwc_dict,
    category_map,
    emotion_word,
    text_column='text',
    resample_unit='W'
):
    """
    Calculates and returns the LIWC percentage metrics for both pre-COVID and main datasets.

    For each post, the LIWC percentage is computed as the percentage of words
    that are associated with the target emotion (using the analyze_emotion function).
    The results are aggregated by resampling (default weekly) and combined into a single
    time series covering both periods.

    Parameters
    ----------
    posts_file : str
        Path to the main posts Parquet file containing 'created_utc' and a text column.
    precovid_posts_file : str
        Path to the pre-COVID posts Parquet file containing 'created_utc' and a text column.
    liwc_dict : dict
        A dictionary where each key is a token and its value is a list of LIWC category codes.
    category_map : dict
        A mapping from LIWC category codes to their descriptive emotion names.
    emotion_word : str
        The target emotion to analyze (e.g., "anger", "joy").
    text_column : str, optional
        The name of the column in the posts files that contains the text to analyze.
        Default is 'text'.
    resample_unit : str, optional
        Pandas resample frequency (default 'W' for weekly).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single row labeled 'LIWC Percentage for <emotion_word>' and
        columns corresponding to date strings (formatted as DD/MM/YYYY), containing the
        average LIWC percentage per resampled period.
    """
    import pandas as pd

    # Note: This function assumes that the `analyze_emotion` function is defined elsewhere.
    # For example:
    #
    # def analyze_emotion(text, liwc_dict, category_map, emotion_word):
    #     tokens = tokenize(text)
    #     total_words = len(tokens)
    #     if total_words == 0:
    #         return 0
    #     count = 0
    #     for token in tokens:
    #         if token in liwc_dict:
    #             for code in liwc_dict[token]:
    #                 if category_map.get(code, "").lower() == emotion_word.lower():
    #                     count += 1
    #     return (count / total_words) * 100

    def process_data(posts_path):
        # Read only the necessary columns: the timestamp and the text column.
        posts = pd.read_parquet(posts_path, columns=['created_utc', text_column])
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
        posts.sort_values('created_datetime', inplace=True)

        # Compute the LIWC percentage for each post.
        posts['liwc_percentage'] = posts[text_column].apply(
            lambda text: analyze_emotion(text, liwc_dict, category_map, emotion_word)
        )

        # Set the datetime column as the index for resampling.
        posts.set_index('created_datetime', inplace=True)

        # Compute the average LIWC percentage for posts in each resampling period.
        liwc_metric = posts['liwc_percentage'].resample(resample_unit).mean()
        return liwc_metric

    # Process the pre-COVID dataset and limit its date range if desired.
    precovid_liwc = process_data(precovid_posts_file)
    precovid_liwc = precovid_liwc['2016-01-01':'2019-12-31']

    # Process the main dataset.
    main_liwc = process_data(posts_file)

    # Combine the two periods into one time series.
    combined_liwc = pd.concat([precovid_liwc, main_liwc])

    # Construct a full date range to ensure continuity.
    full_index = pd.date_range(
        start=combined_liwc.index.min(),
        end=combined_liwc.index.max(),
        freq=resample_unit
    )
    combined_liwc = combined_liwc.reindex(full_index)

    # Interpolate any missing values.
    combined_liwc = combined_liwc.interpolate()

    # Format the result into a DataFrame with the desired row and column labels.
    metrics_table = pd.DataFrame(
        {f'LIWC Percentage for {emotion_word}': combined_liwc.values},
        index=combined_liwc.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def save_timeseries_metrics_for_cities(city_dict):
    
    output_dir = "../metrics_test"
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

        liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
        category_map = load_category_mapping(category_mapping_path)


        affect_metrics = get_liwc_percentage_metrics(
            submissions_path,
            prophet_train_submissions_path,
            liwc_dict,
            category_map,
            "affect",
            text_column='selftext',
            resample_unit='W'
        )
        
        affect_metrics = get_liwc_percentage_metrics(
            submissions_path,
            prophet_train_submissions_path,
            liwc_dict,
            category_map,
            "affect",
            text_column='selftext',
            resample_unit='W'
        )

        combined_metrics = pd.concat([
            post_count_metrics, 
            comment_count_metrics, 
            comment_percentage_metrics, 
            lifespan_metrics, 
            response_metrics, 
            cutoff_response_metrics, 
            affect_metrics
        ], axis=0)

        output_path = os.path.join(output_dir, f"{city_lower}_metrics.parquet")
        combined_metrics.to_parquet(output_path)


# def save_liwc_timeseries_metrics_for_cities(city_dict):
    
#     output_dir = "../liwc_metrics"
#     os.makedirs(output_dir, exist_ok=True)

#     for city_key, city_name in city_dict.items():
#         city_lower = city_key.lower().replace(" ", "")
#         submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
#         comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
#         prophet_train_submissions_path = f"../prophet_train_parquet/{city_lower}_submissions.parquet"
#         prophet_train_comments_path = f"../prophet_train_parquet/{city_lower}_comments.parquet"

#         liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
#         category_map = load_category_mapping(category_mapping_path)

#         affect = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "affect",
#             text_column='selftext',
#             resample_unit='W'
#         )
        
#         posemo = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "posemo",
#             text_column='selftext',
#             resample_unit='W'
#         )

#         negemo = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "negemo",
#             text_column='selftext',
#             resample_unit='W'
#         )

#         anx = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "anx",
#             text_column='selftext',
#             resample_unit='W'
#         )

#         anger = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "anx",
#             text_column='selftext',
#             resample_unit='W'
#         )

#         sad = get_liwc_percentage_metrics(
#             submissions_path,
#             prophet_train_submissions_path,
#             liwc_dict,
#             category_map,
#             "anx",
#             text_column='selftext',
#             resample_unit='W'
#         )

#         combined_metrics = pd.concat([
#             affect,
#             posemo,
#             negemo,
#             anx,
#             anger,
#             sad
#         ], axis=0)

#         output_path = os.path.join(output_dir, f"{city_lower}_liwc_metrics.parquet")
#         combined_metrics.to_parquet(output_path)


def save_liwc_timeseries_metrics_for_cities(city_dict, categories):
    output_dir = "../liwc_metrics"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        prophet_train_submissions_path = f"../prophet_train_parquet/{city_lower}_submissions.parquet"
        prophet_train_comments_path = f"../prophet_train_parquet/{city_lower}_comments.parquet"

        liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
        category_map = load_category_mapping(category_mapping_path)

        metrics_list = []
        for category in categories:
            metric = get_liwc_percentage_metrics(
                submissions_path,
                prophet_train_submissions_path,
                liwc_dict,
                category_map,
                category,
                text_column='selftext',
                resample_unit='W'
            )
            metrics_list.append(metric)

        combined_metrics = pd.concat(metrics_list, axis=0)

        output_path = os.path.join(output_dir, f"{city_lower}_liwc_metrics.parquet")
        combined_metrics.to_parquet(output_path)


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


def aggregate_metrics_by_cluster(metrics_path, clusters_path, output_path, metric, aggregation="mean"):
    """
    Aggregates time series data for each cluster group based on the cluster assignments
    in the clusters.parquet file and the metrics stored in the metrics_path.

    Parameters:
    - metrics_path (str): Path to the folder containing city-specific metrics parquet files.
    - clusters_path (str): Path to the clusters.parquet file containing city-to-cluster assignments.
    - output_path (str): Path to save the aggregated metrics parquet file.
    - aggregation (str): Aggregation method ('mean', 'median', etc.). Default is 'mean'.

    Saves:
    - A single parquet file with aggregated metrics for each cluster.
    """

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


def plot_two_clusters_timeseries(parquet_file_path, save_path=None):
    """
    Reads a parquet file that contains:
      - An index column named '_index_level_0' representing timestamps
      - Two columns: 'cluster_0' and 'cluster_1' for time series data
    Plots both time series on the same figure, and optionally saves the plot.

    :param parquet_file_path: Path to the Parquet file.
    :param save_path: (Optional) Path to save the generated plot. If None, 
                      the plot will not be saved.
    """
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


def predict_and_plot_cluster_aggregates(aggregated_data_path, output_plot_path):
    """
    Trains a time series model on all data before January 1, 2020,
    and plots predictions against actual observed aggregate scores from January 1, 2020 onwards.

    Parameters:
    - aggregated_data_path (str): Path to the aggregated cluster metrics parquet file.
    - output_plot_path (str): Path to save the prediction and actual comparison plot.
    """
    data = pd.read_parquet(aggregated_data_path)
    data.index = pd.to_datetime(data.index)

    split_date = pd.Timestamp("2020-01-01")

    plt.figure(figsize=(12, 6))
    for cluster in data.columns:
        print(f"Processing {cluster}...")

        cluster_data = data[[cluster]].reset_index()
        cluster_data.columns = ["ds", "y"]

        train_data = cluster_data[cluster_data["ds"] < split_date]
        test_data = cluster_data[cluster_data["ds"] >= split_date]

        model = Prophet(weekly_seasonality=True)
        model.fit(train_data)

        future = model.make_future_dataframe(periods=len(test_data), freq="W")
        forecast = model.predict(future)

        predicted = forecast[forecast["ds"].isin(test_data["ds"])]["yhat"]
        mae = mean_absolute_error(test_data["y"], predicted)
        rmse = sqrt(mean_squared_error(test_data["y"], predicted))
        print(f"Forecast MAE for {cluster}: {mae:.2f}")
        print(f"Forecast RMSE for {cluster}: {rmse:.2f}")

        plt.plot(cluster_data["ds"], cluster_data["y"], label=f"Observed {cluster}", alpha=0.6)
        plt.plot(forecast["ds"], forecast["yhat"], linestyle="--", label=f"Predicted {cluster}", alpha=0.8)

        plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01" if cluster == data.columns[0] else "")

    plt.title("Time Series Prediction from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel("Aggregate Metric")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

    print(f"Prediction plot saved to {output_plot_path}")


def predict_and_plot_time_series(data_path, metric_name, output_plot_path):
    """
    Fits a Prophet model on data before January 1, 2020, for a specified metric,
    and plots the forecast against actual data from January 1, 2020, onwards.

    Parameters:
    - data_path (str): Path to the city-level parquet file containing the *wide* time series data 
                       (date columns, metric rows).
    - metric_name (str): Name of the metric (one of the columns after transposing) to forecast.
    - output_plot_path (str): Path to save the prediction/actual comparison plot.
    """

    data = pd.read_parquet(data_path)

    data.columns = pd.to_datetime(data.columns, format="%d/%m/%Y")

    data = data.T 

    if metric_name not in data.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found. "
            f"Available metrics: {list(data.columns)}"
        )

    data = data[[metric_name]]

    data.rename(columns={metric_name: 'y'}, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'ds'}, inplace=True)

    split_date = pd.Timestamp("2020-01-01")
    train_data = data[data['ds'] < split_date].copy()
    test_data  = data[data['ds'] >= split_date].copy()

    model = Prophet(weekly_seasonality=True)
    model.fit(train_data)

    future = model.make_future_dataframe(periods=len(test_data), freq="W")
    forecast = model.predict(future)

    forecast_2020plus = forecast[forecast['ds'] >= split_date]

    y_true = test_data['y'].values
    y_pred = forecast_2020plus['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print(f"Forecast MAE for {metric_name}: {mae:.2f}")
    print(f"Forecast RMSE for {metric_name}: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label=f"Observed {metric_name}", alpha=0.6)
    plt.plot(forecast_2020plus['ds'], forecast_2020plus['yhat'],
             linestyle="--", label=f"Predicted {metric_name}", alpha=0.8)

    plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01")

    plt.title(f"Time Series Prediction for {metric_name} from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path) or ".", exist_ok=True)
    plt.savefig(output_plot_path)
    plt.show()
    print(f"Prediction plot saved to {output_plot_path}")


def train_and_validate_prophet(train_df, test_df, freq='D'):
    """
    Train a Prophet model on 'train_df' (with columns ['ds', 'y']).
    Then predict over the test period and compute MAE/RMSE on 'test_df'.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training slice with columns ['ds', 'y'].
    test_df : pd.DataFrame
        Test slice with columns ['ds', 'y'].
    freq : str
        Frequency for future dataframe (e.g. 'D', 'W').

    Returns
    -------
    model : Prophet
        Fitted Prophet model.
    forecast : pd.DataFrame
        Forecast for the entire (train + test) date range.
    mae : float
        Mean absolute error on test set.
    rmse : float
        Root mean squared error on test set.
    """
    model = Prophet(weekly_seasonality=True)
    model.fit(train_df)

    last_train_date = train_df['ds'].max()
    last_test_date = test_df['ds'].max()

    horizon_days = (last_test_date - last_train_date).days

    future = model.make_future_dataframe(periods=horizon_days, freq=freq)
    forecast = model.predict(future)

    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
    mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
    rmse = sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))

    return model, forecast, mae, rmse


def predict_and_plot_single_df(
    aggregated_data_path,
    output_plot_path,
    train_start='2019-07-01',
    train_end='2019-09-30',
    test_start='2019-10-01',
    test_end='2019-12-31',
    post_test_start='2020-01-01',
    freq='D'
):
    """
    1) Loads a single DataFrame from 'aggregated_data_path'.
    2) For each cluster (column):
       - Slices the DataFrame into train (2019-07-01 -> 2019-09-30)
         and test  (2019-10-01 -> 2019-12-31)
       - Trains and validates Prophet (computes MAE, RMSE)
       - Forecasts after 2020-01-01
       - Plots observed vs. predicted

    Parameters
    ----------
    aggregated_data_path : str
        Path to the single parquet file with all data.
    output_plot_path : str
        Where to save the final plot.
    train_start, train_end, test_start, test_end : str
        Date ranges for train and test.
    post_test_start : str
        Start date for final “observed vs. predicted” plotting.
    freq : str
        Frequency for Prophet (e.g., 'D' or 'W').
    """
    data = pd.read_parquet(aggregated_data_path)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts   = pd.Timestamp(train_end)
    test_start_ts  = pd.Timestamp(test_start)
    test_end_ts    = pd.Timestamp(test_end)
    post_test_ts   = pd.Timestamp(post_test_start)

    clusters = data.columns
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes] 

    metrics = {
        'cluster': [],
        'MAE': [],
        'RMSE': []
    }

    for i, cluster in enumerate(clusters):
        ax = axes[i]

        cluster_df = data[[cluster]].reset_index()
        cluster_df.columns = ['ds', 'y']

        train_mask = (cluster_df['ds'] >= train_start_ts) & (cluster_df['ds'] <= train_end_ts)
        test_mask =  (cluster_df['ds'] >= test_start_ts)  & (cluster_df['ds'] <= test_end_ts)

        train_data = cluster_df.loc[train_mask]
        test_data  = cluster_df.loc[test_mask]

        try:
            model, in_sample_forecast, mae, rmse = train_and_validate_prophet(train_data, test_data, freq=freq)
            metrics['cluster'].append(cluster)
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
        except ValueError as e:
            print(f"Skipping cluster '{cluster}' due to error: {e}")
            metrics['cluster'].append(cluster)
            metrics['MAE'].append(None)
            metrics['RMSE'].append(None)
            continue

        print(f"\nCluster '{cluster}': MAE={mae:.2f}, RMSE={rmse:.2f}")

        last_test_date = test_end_ts
        max_date = cluster_df['ds'].max()
        days_to_forecast = (max_date - last_test_date).days

        if days_to_forecast > 0:
            future_df = model.make_future_dataframe(periods=days_to_forecast, freq=freq)
            final_forecast = model.predict(future_df)
        else:
            final_forecast = in_sample_forecast

        ax.plot(cluster_df['ds'], cluster_df['y'], label='Observed', color='black', alpha=0.6)

        ax.plot(in_sample_forecast['ds'], in_sample_forecast['yhat'], 
                color='blue', linestyle='--', label='Train+Test Prediction')

        ax.plot(final_forecast['ds'], final_forecast['yhat'], 
                color='red', linestyle='--', label='Post-Test Prediction')
        
        ax.axvline(train_end_ts, color='orange', linestyle='--', label='Train End')
        ax.axvline(test_end_ts,  color='green',  linestyle='--', label='Test End')

        ax.set_xlim([train_start_ts - pd.Timedelta(days=5), max_date + pd.Timedelta(days=5)])

        ax.set_title(f"Cluster: {cluster}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Metric")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_plot_path}")

    metrics_df = pd.DataFrame(metrics)
    print("Validation Metrics:")
    print(metrics_df)
    return metrics_df


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

    city_populations = {
        "albuquerque": 564559,
        "anchorage": 291247,
        "arlington": 394266,
        "atlanta": 498715,
        "aurora": 386261,
        "austin": 961855,
        "bakersfield": 403455,
        "baltimore": 585708,
        "batonrouge": 227470,
        "boise": 235684,
        "boston": 675647,
        "buffalo": 278349,
        "charlotte": 874579,
        "chicago": 2746388,
        "cincinnati": 309317,
        "cleveland": 372624,
        "coloradosprings": 478961,
        "columbus": 905748,
        "corpuschristi": 317863,
        "dallas": 1304379,
        "denver": 715522,
        "detroit": 639111,
        "elpaso": 678815,
        "fortwayne": 263886,
        "fortworth": 918915,
        "fremont": 230504,
        "fresno": 542107,
        # "frisco": 200509,
        "glendale": 248325,
        "honolulu": 345510,
        "houston": 2304580,
        # "huntsville": 215006,
        "indianapolis": 887642,
        "irvine": 307670,
        "jacksonville": 949611,
        "jerseycity": 292449,
        "kansascity": 508090,
        "laredo": 255205,
        "lasvegas": 641903,
        "lexington": 322570,
        "lincoln": 291082,
        "longbeach": 466742,
        "losangeles": 3898747,
        "louisville": 633045,
        "lubbock": 257141,
        "madison": 269840,
        "memphis": 633104,
        "miami": 442241,
        "milwaukee": 577222,
        "minneapolis": 429954,
        "nashville": 689447,
        "newark": 311549,
        "neworleans": 383997,
        "norfolk": 238005,
        "newyorkcity": 8804190,
        "oakland": 440646,
        "oklahomacity": 681054,
        "omaha": 486051,
        "orlando": 307573,
        "philadelphia": 1603797,
        "pittsburgh": 302971,
        "plano": 285494,
        "portland": 652503,
        "reno": 264165,
        "richmond": 226610,
        "riverside": 314998,
        "sacramento": 524943,
        "saintpaul": 311527,
        "sanantonio": 1434625,
        "sandiego": 1386932,
        "sanfrancisco": 873965,
        "sanjose": 1013240,
        "santaclarita": 228673,
        "scottsdale": 241361,
        "seattle": 737015,
        "spokane": 228989,
        "stlouis": 301578,
        "stockton": 320804,
        "stpetersburg": 258308,
        "tampa": 384959,
        "toledo": 270871,
        "tucson": 542629,
        "tulsa": 413066,
        "virginiabeach": 459470,
        "washingtondc": 689545,
        "wichita": 397532,
        "winstonsalem": 249545
    }

    city_users = {
        "albuquerque": 12769,
        "anchorage": 6762,
        "arlington": 2248,
        "atlanta": 35391,
        "aurora": 1807,
        "austin": 74510,
        "bakersfield": 5685,
        "baltimore": 19488,
        "batonrouge": 7130,
        "boise": 8078,
        "boston": 56394,
        "buffalo": 14016,
        "charlotte": 23045,
        "chicago": 61158,
        "cincinnati": 21303,
        "cleveland": 18142,
        "coloradosprings": 14216,
        "columbus": 37921,
        "corpuschristi": 1244,
        "dallas": 45442,
        "denver": 59148,
        "detroit": 20517,
        "elpaso": 5093,
        "fortwayne": 4829,
        "fortworth": 9997,
        "fremont": 1730,
        "fresno": 6718,
        # "frisco": 328,
        "glendale": 1174,
        "honolulu": 2134,
        "houston": 59999,
        # "huntsville": 2875,
        "indianapolis": 20154,
        "irvine": 2781,
        "jacksonville": 11124,
        "jerseycity": 9062,
        "kansascity": 27450,
        "laredo": 974,
        "lasvegas": 21302,
        "lexington": 9082,
        "lincoln": 7074,
        "longbeach": 10129,
        "losangeles": 95178,
        "louisville": 18814,
        "lubbock": 4591,
        "madison": 21012,
        "memphis": 11187,
        "miami": 20742,
        "milwaukee": 17366,
        "minneapolis": 52404,
        "nashville": 32402,
        "newark": 1543,
        "neworleans": 25394,
        "norfolk": 4723,
        "newyorkcity": 25297,
        "oakland": 13016,
        "oklahomacity": 10042,
        "omaha": 15206,
        "orlando": 21906,
        "philadelphia": 47429,
        "pittsburgh": 31479,
        "plano": 5201,
        "portland": 75076,
        "reno": 12746,
        "richmond": 21913,
        "riverside": 2278,
        "richmond": 21913,
        "sacramento": 31786,
        "saintpaul": 2252,
        "sanantonio": 24169,
        "sandiego": 49866,
        "sanfrancisco": 46693,
        "sanjose": 22342,
        "santaclarita": 1895,
        "scottsdale": 3345,
        "seattle": 88101,
        "spokane": 11711,
        "stlouis": 26992,
        "stockton": 1318,
        "stpetersburg": 8755,
        "tampa": 20280,
        "toledo": 4112,
        "tucson": 13552,
        "tulsa": 13474,
        "virginiabeach": 5177,
        "washingtondc": 39618,
        "wichita": 6333,
        "winstonsalem": 3890,
    }
    
    comment_percentage_cluster_0 = ['honolulu', 'chicago', 'newyorkcity', 'atlanta', 'laredo', 'glendale', 'aurora', 'newark', 'miami', 'corpuschristi', 'boise', 'losangeles', 'scottsdale', 'riverside', 'santaclarita', 'saintpaul', 'stockton', 'fortworth', 'houston', 'philadelphia']

    precovid_start_str = '2019-07-01'
    precovid_end_str = '2019-12-31'

    prophet_train_start_str = '2016-01-01'
    prophet_train_end_str = '2019-12-31'

    start_date_str = '2020-01-01'
    end_date_str = '2021-12-31'

    nyc_submissions_path = f"../covid_data_parquet/newyorkcity_submissions.parquet"
    nyc_comments_path = f"../covid_data_parquet/newyorkcity_comments.parquet"
    nyc_precovid_submissions_path = f"../precovid_data_parquet/newyorkcity_submissions.parquet"
    nyc_precovid_comments_path = f"../precovid_data_parquet/newyorkcity_comments.parquet"

    liwc_dictionary_path = '../liwc/LIWC2007_English080730.dic'
    category_mapping_path = '../liwc/LIWC2007_Categories.txt'

    categories = [
        "affect",
        "posemo",
        "negemo",
        "anx",
        "anger",
        "sad",
        "swear",
        "social",
        "family",
        "humans",
        "incl",
        "excl",
        "cogmech",
        "insight",
        "cause",
        "discrep",
        "tentat",
        "certain"
        # "percept",
        # "see",
        # "hear",
        # "feel",
        # "bio",
        # "body",
        # "health",
        # "ingest",
        # "relativ",
        # "motion",
        # "space",
        # "time",
        # "work",
        # "achieve",
        # "money"
    ]


    # df = pd.read_parquet("../response_cutoff_clusters.parquet")
    # cluster = [df["City"][i] for i,j in enumerate(df["Cluster"]) if j == 1]
    # print(cluster, len(cluster))

    # for city in cities:
    #     raw_json_path = f"../raw_json/{city}_submissions.json"
    #     output_path = f"../prophet_train_json/{city}_submissions.json"
    #     print(city, "submission")
    #     filter_and_extract_json(raw_json_path, output_path, "submission", prophet_train_start_str, prophet_train_end_str)

    #     raw_json_path = f"../raw_json/{city}_comments.json"
    #     output_path = f"../prophet_train_json/{city}_comments.json"
    #     print(city, "comment")
    #     filter_and_extract_json(raw_json_path, output_path, "comment", prophet_train_start_str, prophet_train_end_str)


    # for city in cities:
    #     in_path_comments = f"../prophet_train_json/{city}_comments.json"
    #     out_path_comments = f"../prophet_train_parquet/{city}_comments.parquet"
    #     print(city, "comments parquet")
    #     json_file_to_parquet(in_path_comments, out_path_comments)

    #     in_path_submissions = f"../prophet_train_json/{city}_submissions.json"
    #     out_path_submissions = f"../prophet_train_parquet/{city}_submissions.parquet"
    #     print(city, "submissions parquet")
    #     json_file_to_parquet(in_path_submissions, out_path_submissions)

    # for city in cities:
    #     in_path_comments = f"../prophet_train_parquet/{city}_comments.parquet"
    #     out_path_comments = f"../prophet_train_parquet/{city}_comments.parquet"
    #     print(city, "bot comments")
    #     remove_bot_entries_parquet(in_path_comments, out_path_comments)
        
    #     in_path_submissions = f"../prophet_train_parquet/{city}_submissions.parquet"
    #     out_path_submissions = f"../prophet_train_parquet/{city}_submissions.parquet"
    #     print(city, "bot submissions")
    #     remove_bot_entries_parquet(in_path_submissions, out_path_submissions)
    
    # remove_automoderator_data(cities, source_folder="../prophet_train_parquet", target_folder="../prophet_train_parquet")

    save_liwc_timeseries_metrics_for_cities(city_dict, categories)

    # plot_two_clusters_timeseries("../aggregated_cluster_metrics.parquet", "../lifespan_clusters.png")

    # predict_and_plot_single_df(
    #     aggregated_data_path="../aggregated_cluster_metrics.parquet",
    #     output_plot_path="../cluster_predictions_with_validation.png"
    # )

    # predict_and_plot_cluster_aggregates(
    #     aggregated_data_path="../aggregated_response_times.parquet",
    #     output_plot_path="../response_cluster_forecast.png"
    # )

    # predict_and_plot_cluster_aggregates(
    #     aggregated_data_path="../aggregated_response_cutoff.parquet",
    #     output_plot_path="../response_cutoff_cluster_forecast.png"
    # )

    # predict_and_plot_time_series("../metrics/losangeles_metrics.parquet", "Average Response Time (Minutes)", "../losangeles_response_prophet.png")

    # predict_and_plot_time_series("../metrics/houston_metrics.parquet", "Average Response Time with Cutoff (Minutes)", "../houston_response_cutoff_prophet.png")

    # cluster_cities_with_dtw_pyclustering(
    #     city_files_folder="../metrics",
    #     output_file="../comments_percentage_subclusters.parquet",
    #     metric_name="Percentage of Posts with Comments",
    #     resample_unit="W",
    #     max_clusters=10,
    #     n_jobs=-1,
    #     selected_cities = comment_percentage_cluster_0
    # )

    # cluster_cities_with_dtw_pyclustering(
    #     city_files_folder="../metrics",
    #     output_file="../response_cutoff_clusters.parquet",
    #     metric_name="Average Response Time with Cutoff (Minutes)",
    #     resample_unit="W",
    #     max_clusters=10,
    #     n_jobs=-1,
    #     selected_cities=None
    # )

    # aggregate_metrics_by_cluster(
    #     metrics_path="../metrics",
    #     clusters_path="../comments_percentage_clusters.parquet",
    #     output_path="../aggregated_comment_percentage.parquet",
    #     metric="Percentage of Posts with Comments",
    #     aggregation="mean"
    # )

    # aggregate_metrics_by_cluster(
    #     metrics_path="../metrics",
    #     clusters_path="../response_cutoff_clusters.parquet",
    #     output_path="../aggregated_response_cutoff.parquet",
    #     metric="Average Response Time with Cutoff (Minutes)",
    #     aggregation="mean"
    # )

    # cluster_cities_with_dtw_pyclustering(
    #     city_files_folder="../metrics",
    #     output_file="../response_clusters.parquet",
    #     metric_name="Average Response Time (Minutes)",
    #     resample_unit="W",
    #     max_clusters=10,
    #     n_jobs=-1
    # )

    # plot_comment_percentage_time_window(
    #     nyc_submissions_path,
    #     nyc_comments_path,
    #     nyc_precovid_submissions_path,
    #     nyc_precovid_comments_path,
    #     city="newyorkcity",
    #     resample_unit='W',  # 'W' for weekly, 'D' for daily, 'M' for monthly, etc.
    #     save_plot=True,
    #     plot_path=f'../newyorkcity_comment_percentage_same_week.png',
    #     display_plot=False
    # )

    # plot_post_count(
    #     nyc_submissions_path, 
    #     nyc_comments_path,
    #     nyc_precovid_submissions_path,
    #     nyc_precovid_comments_path, 
    #     city="newyorkcity", 
    #     resample_unit='W', 
    #     save_plot=True, 
    #     plot_path=f'../newyorkcity_post_count.png', 
    #     display_plot=False
    # )

    # plot_response_times_with_cutoff(
    #     nyc_submissions_path,
    #     nyc_comments_path,
    #     city="newyorkcity",
    #     time_unit='W',
    #     time_diff_unit='hours',
    #     save_plot=True,
    #     plot_path='../newyorkcity_average_response_times_cutoff.png',
    #     display_plot=False
    # )

    # generate_graphs_for_cities(city_dict)

    # remove_automoderator_data(cities, source_folder="../precovid_data_parquet", target_folder="../precovid_data_parquet_2")

    # compute_new_and_returning_users("../precovid_data_parquet", "../covid_data_parquet")

    # plot_post_sentiment('../covid_data_parquet/newyorkcity_submissions.parquet', 'New York City')

    # plot_comment_percentage('../covid_data_parquet/newyorkcity_submissions.parquet', '../covid_data_parquet/newyorkcity_comments.parquet')
    
    # plot_response_times('../covid_data_parquet/newyorkcity_submissions.parquet', '../covid_data_parquet/newyorkcity_comments.parquet', time_unit='W', time_diff_unit='minutes')

    # plot_post_lifespan_timeseries(
    #     posts_parquet_path='../covid_data_parquet/newyorkcity_submissions.parquet',
    #     comments_parquet_path='../covid_data_parquet/newyorkcity_comments.parquet',
    #     aggregation='weekly',      # Options: 'daily', 'weekly', 'monthly', 'yearly'
    #     units='hours',              # Options: 'seconds', 'minutes', 'hours', 'days'
    #     statistic='mean',           # Options: 'mean', 'median', 'count'
    #     title='New York City Weekly Average Post Lifespan (Excluding Single-Comment Posts)',
    #     xlabel='Week',
    #     ylabel='Average Lifespan (Hours)',
    #     figsize=(16, 8),
    #     return_data=True,           # Capture the aggregated data
    #     log_scale=False,            # No log scale
    #     verbose=False                # Enable verbose logging
    # )


    # plot_comments_distribution("../covid_data_parquet/newyorkcity_submissions.parquet", "New York City Comments Distribution", "Log No. Comments", "Log No. Posts", True, True)

    # user_population_spearman(city_users, city_populations)

    # convert_json_to_array("../raw_json", "../raw_json_array")
    
    # convert_jsons_to_parquets_efficient("../raw_json", "../raw_parquet", chunk_size=50000)

    # process_city_files(cities)

    # for city in cities:
    #     in_path_comments = f"../covid_data_json/{city}_comments.json"
    #     out_path_comments = f"../covid_data_parquet/{city}_comments.parquet"
    #     print(city, "comments during")
    #     json_file_to_parquet(in_path_comments, out_path_comments)

    #     in_path_submissions = f"../covid_data_json/{city}_submissions.json"
    #     out_path_submissions = f"../covid_data_parquet/{city}_submissions.parquet"
    #     print(city, "submissions during")
    #     json_file_to_parquet(in_path_submissions, out_path_submissions)

    #     in_path_comments_pre = f"../precovid_data_json/{city}_comments.json"
    #     out_path_comments_pre = f"../precovid_data_parquet/{city}_comments.parquet"
    #     print(city, "comments pre")
    #     json_file_to_parquet(in_path_comments_pre, out_path_comments_pre)

    #     in_path_submissions_pre = f"../precovid_data_json/{city}_submissions.json"
    #     out_path_submissions_pre = f"../precovid_data_parquet/{city}_submissions.parquet"
    #     print(city, "submissions pre")
    #     json_file_to_parquet(in_path_submissions_pre, out_path_submissions_pre)


    # pre_covid_path = "../precovid_data_parquet"
    # during_covid_path = "../covid_data_parquet"
    # output_path = "new_and_returning_users.parquet"
    
    # user_stats_df = compute_new_and_returning_users(pre_covid_path, during_covid_path, output_path)
    # print(user_stats_df)

    # for city in cities:
    #     raw_json_path = f"../raw_json/{city}_submissions.json"
    #     output_path = f"../precovid_json/{city}_submissions.json"
    #     print(city, "submission")
    #     filter_and_extract_json(raw_json_path, output_path, "submission", precovid_start_str, precovid_end_str)

    #     raw_json_path = f"../raw_json/{city}_comments.json"
    #     output_path = f"../precovid_json/{city}_comments.json"
    #     print(city, "comment")
    #     filter_and_extract_json(raw_json_path, output_path, "comment", precovid_start_str, precovid_end_str)

    # convert_jsons_to_parquets("../raw_json", "../raw_parquet")


    # extract_data_between_dates("../data/newyorkcity/comments/processed/comments.json", "../data/newyorkcity/comments/processed/comments2.json", start_date_str, end_date_str)
    # filter_comments("../data/newyorkcity/comments/processed/comments2.json", "../data/newyorkcity/comments/processed/comments3.json")

    # extract_data_between_dates("../data/newyorkcity/submissions/processed/submissions.json", "../data/newyorkcity/submissions/processed/submissions2.json", start_date_str, end_date_str)
    # filter_submissions("../data/newyorkcity/submissions/processed/submissions2.json", "../data/newyorkcity/submissions/processed/submissions3.json")

    # for city in cities:
    #     print(f"\"{city}\":", str(count_unique_authors(f"../data/{city}/comments/processed/comments.parquet"))+",")
        # plot_entries_per_week(f"../data/{city}/comments/processed/comments.json", f"../data/{city}/comments/processed", "comments")
        # plot_entries_per_week(f"../data/{city}/submissions/processed/submissions.json", f"../data/{city}/submissions/processed", "submissions")

        # plot_entries_per_month(f"../data/{city}/comments/processed/comments.json", f"../data/{city}/comments/processed", "comments")
        # plot_entries_per_month(f"../data/{city}/submissions/processed/submissions.json", f"../data/{city}/submissions/processed", "submissions")

    # calculate_spearman_correlation(cities, city_users, "comments")
    # calculate_spearman_correlation(cities, city_users, "submissions")
    # calculate_spearman_correlation(cities, city_populations, "submissions")

    # for city in cities:
    #     print("\n"+city)
    #     lowest_entries_per_week(f"../data/{city}/comments/processed/comments.json", f"../data/{city}/comments/processed", "comment")
    #     lowest_entries_per_week(f"../data/{city}/submissions/processed/submissions.json", f"../data/{city}/submissions/processed", "submission")

    #     remove_bot_entries_parquet(f"../data/{city}/comments/processed/comments.parquet", f"../data/{city}/comments/processed/comments.parquet")
    #     remove_bot_entries_parquet(f"../data/{city}/submissions/processed/submissions.parquet", f"../data/{city}/submissions/processed/submissions.parquet")

    # extract_data_between_dates("../data/stpetersburg/comments/processed/comments.json", "../data/stpetersburg/comments/processed/comments2.json", start_date_str, end_date_str)
    # filter_comments("../data/stpetersburg/comments/processed/comments2.json", "../data/stpetersburg/comments/processed/comments3.json")

    # extract_data_between_dates("../data/stpetersburg/submissions/processed/submissions.json", "../data/stpetersburg/submissions/processed/submissions2.json", start_date_str, end_date_str)
    # filter_submissions("../data/stpetersburg/submissions/processed/submissions2.json", "../data/stpetersburg/submissions/processed/submissions3.json")

    # for city in cities:
    #     json_file_to_parquet(f"../data/{city}/comments/processed/comments.json", f"../data/{city}/comments/processed/comments.parquet")
    #     json_file_to_parquet(f"../data/{city}/submissions/processed/submissions.json", f"../data/{city}/submissions/processed/submissions.parquet")


    # if len(sys.argv) != 4:
    #     print("Usage: python3 main.py <input_directory> <output_directory> <dataset_type>")
    #     sys.exit(1)

    # input_directory = sys.argv[1]
    # output_directory = sys.argv[2]
    # dataset_type = sys.argv[3].lower()  # Should be 'comments' or 'submissions'

    # if dataset_type not in ['comments', 'submissions']:
    #     print("Invalid dataset type. Please specify 'comments' or 'submissions'.")
    #     sys.exit(1)

    # os.makedirs(output_directory, exist_ok=True)

    # start_date_str = '2020-01-01'
    # end_date_str = '2021-12-31'

    # for filename in os.listdir(input_directory):
    #     input_file_path = os.path.join(input_directory, filename)
    #     if os.path.isfile(input_file_path) and filename.endswith('.json'):
    #         print(f"\nProcessing file: {input_file_path}")

    #         temp_output_path = os.path.join(output_directory, 'temp_' + filename)
    #         final_output_path = os.path.join(output_directory, filename)

    #         # Step 1: Extract data between dates
    #         print("Extracting data between dates...")
    #         success = extract_data_between_dates(input_file_path, temp_output_path, start_date_str, end_date_str)
    #         if not success:
    #             print(f"Skipping file due to extraction error: {filename}")
    #             continue

    #         # # Step 2: Count the number of entries
    #         # print("Counting entries...")
    #         # num_entries = num_lines(temp_output_path)
    #         # print(f"The JSON file {filename} contains {num_entries} entries after date filtering.")

    #         # if num_entries == 0:
    #         #     print(f"No entries found in date range for file: {filename}")
    #         #     os.remove(temp_output_path)
    #         #     continue

    #         # Step 3: Filter data to keep specified fields
    #         print("Filtering data...")
    #         if dataset_type == 'submissions':
    #             success = filter_submissions(temp_output_path, final_output_path)
    #         elif dataset_type == 'comments':
    #             success = filter_comments(temp_output_path, final_output_path)
    #         else:
    #             print(f"Unknown dataset type: {dataset_type}")
    #             os.remove(temp_output_path)
    #             continue

    #         if not success:
    #             print(f"Skipping file due to filtering error: {filename}")
    #             os.remove(temp_output_path)
    #             continue

    #         # Remove the temporary file
    #         os.remove(temp_output_path)

    #         # # Step 4: Count unique authors
    #         # print("Counting unique authors...")
    #         # num_unique_authors = count_unique_authors(final_output_path)
    #         # print(f"Number of unique authors in {filename}: {num_unique_authors}")

    #         # Step 5: Plot data per week and save the plot
    #         print("Plotting data per week...")
    #         plot_entries_per_week(final_output_path, output_directory, dataset_type)

    # print("\nData processing completed.")