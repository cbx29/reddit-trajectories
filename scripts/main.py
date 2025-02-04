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
    # Include the entire end date
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

    # Write the filtered data to the output file as a JSON array
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
    # Read JSON file into DataFrame
    df = pd.read_json(json_file_path)
    
    # Save DataFrame to Parquet file
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Parquet file saved to {output_file}")


def remove_bot_entries_parquet(input_file, output_file):
    # Regular expression pattern for detecting bot authors
    bot_pattern = re.compile(r'(?:^|\b|_)(bot)(?:$|\b|_)', re.IGNORECASE)
    
    try:
        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error reading Parquet file '{input_file}': {e}")
        return False

    if df.empty:
        print(f"No data to filter in file: '{input_file}'")
        return False

    # Identify bot authors using the regex pattern
    # Create a boolean mask where True indicates entries to remove
    is_bot_author = df['author'].str.contains(bot_pattern)
    
    # Collect the unique bot authors
    bot_authors = df.loc[is_bot_author, 'author'].unique()

    # Filter out entries from bot authors
    df_filtered = df.loc[~is_bot_author].copy()

    # Optionally, print the bot authors that were found and removed
    # if len(bot_authors) > 0:
    #     print("Removed entries from the following bot authors:")
    #     for author in sorted(bot_authors):
    #         print(author)
    # else:
    #     print("No bot authors found.")

    try:
        # Save the filtered DataFrame to a new Parquet file
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

    # Load the parquet file
    for city in cities:
        data = pd.read_parquet(f"../data/{city}/{type}/processed/{type}.parquet")

        # Count the number of comments/submissions in the dataset
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
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Check if 'author' column exists
    if 'author' not in df.columns:
        raise ValueError("The 'author' field is not present in the Parquet file.")

    # Calculate the number of unique authors
    unique_authors_count = df['author'].nunique()

    return unique_authors_count


def process_city_files(city_names):
    raw_dir = "../raw"
    output_dir = "../raw_json"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for city in city_names:
        city_raw_dir = os.path.join(raw_dir, city)
        if not os.path.exists(city_raw_dir):
            print(f"Directory for city '{city}' not found in {raw_dir}. Skipping...")
            continue
        
        # Define file paths
        submissions_zst = os.path.join(city_raw_dir, f"{city}_submissions.zst")
        comments_zst = os.path.join(city_raw_dir, f"{city}_comments.zst")
        submissions_json = os.path.join(output_dir, f"{city}_submissions.json")
        comments_json = os.path.join(output_dir, f"{city}_comments.json")
        
        # Process submissions.zst
        if os.path.exists(submissions_zst):
            print(f"Processing {submissions_zst}...")
            subprocess.run(
                f"zstd -d {submissions_zst} -o {submissions_json}", 
                shell=True, 
                check=True
            )
        else:
            print(f"File {submissions_zst} not found. Skipping...")
        
        # Process comments.zst
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
    # Ensure the output directory exists
    os.makedirs(parquet_dir, exist_ok=True)
    
    # Iterate over all files in the JSON directory
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_file_path = os.path.join(json_dir, json_file)
            parquet_file_name = json_file.replace(".json", ".parquet")
            parquet_file_path = os.path.join(parquet_dir, parquet_file_name)
            
            try:
                print(f"Processing {json_file_path} into {parquet_file_path} in chunks...")
                
                # Process the JSON file in chunks
                chunks = pd.read_json(
                    json_file_path,
                    lines=True,
                    chunksize=chunk_size  # Read in chunks
                )
                
                # Write each chunk to the Parquet file
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # Write the first chunk (creates the file)
                        chunk.to_parquet(parquet_file_path, engine='pyarrow', index=False)
                    else:
                        # Append subsequent chunks
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
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in os.listdir(input_dir):
        if json_file.endswith(".json"):
            input_file_path = os.path.join(input_dir, json_file)
            output_file_path = os.path.join(output_dir, json_file)
            
            try:
                print(f"Converting {input_file_path} to JSON array format...")
                
                # Read line-delimited JSON file
                with open(input_file_path, "r") as infile:
                    json_objects = [json.loads(line) for line in infile]
                
                # Write as a JSON array
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
    # Define fields to keep based on type
    fields_to_keep = {
        "submission": {"author", "author_created_utc", "created_utc", "num_comments", "selftext", "title", "id", "ups", "downs", "score"},
        "comment": {"author", "body", "created_utc", "id", "parent_id", "subreddit", "ups", "downs", "score", "distinguished"}
    }

    if type not in fields_to_keep:
        raise ValueError(f"Invalid type '{type}'. Must be 'comment' or 'submission'.")

    # Select fields for the specified type
    selected_fields = fields_to_keep[type]

    # Convert date strings to datetime objects if provided
    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    else:
        start_date = end_date = None

    filtered_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)  # Handles NDJSON
                # Filter by date range if applicable
                if start_date and end_date:
                    created_utc = entry.get('created_utc')
                    if created_utc is not None:
                        if isinstance(created_utc, str):
                            created_utc = int(float(created_utc))
                        created_date = datetime.utcfromtimestamp(created_utc)
                        if not (start_date <= created_date <= end_date):
                            continue

                # Filter by fields
                entry = {k: v for k, v in entry.items() if k in selected_fields}
                filtered_data.append(entry)

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {line_num} in {input_file}: {e}")
                continue

    if not filtered_data:
        print(f"No data found matching criteria in file: {input_file}")
        return False

    # Write the filtered data to the output file as a JSON array
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


def user_population_spearman(city_users, city_population):
    # Step 1: Ensure both dictionaries have the same set of cities
    common_cities = set(city_populations.keys()) & set(city_users.keys())

    # Optional: Check for missing cities
    missing_in_users = set(city_populations.keys()) - set(city_users.keys())
    missing_in_populations = set(city_users.keys()) - set(city_populations.keys())

    if missing_in_users:
        print(f"Cities missing in users data: {missing_in_users}")
    if missing_in_populations:
        print(f"Cities missing in population data: {missing_in_populations}")

    # Proceeding with common cities
    populations = []
    users = []

    for city in common_cities:
        populations.append(city_populations[city])
        users.append(city_users[city])

    # Step 2: Calculate Spearman Correlation
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
    
    # Helper function to extract city names from filenames
    def extract_city_names(folder: str) -> Set[str]:
        city_names = set()
        for file in os.listdir(folder):
            if file.endswith('.parquet'):
                # Assuming filenames are in the format {city}_comments.parquet or {city}_submissions.parquet
                city = file.rsplit('_', 1)[0]
                city_names.add(city)
        return city_names

    # Extract all unique cities from the pre-COVID and during-COVID folders
    pre_cities = extract_city_names(pre_covid_folder)
    during_cities = extract_city_names(during_covid_folder)
    
    # Union of cities present in either pre or during COVID folders
    all_cities = pre_cities.union(during_cities)
    
    results = []

    for city in sorted(all_cities):
        # Define file paths
        pre_comments_path = os.path.join(pre_covid_folder, f"{city}_comments.parquet")
        pre_submissions_path = os.path.join(pre_covid_folder, f"{city}_submissions.parquet")
        during_comments_path = os.path.join(during_covid_folder, f"{city}_comments.parquet")
        during_submissions_path = os.path.join(during_covid_folder, f"{city}_submissions.parquet")
        
        # Initialize sets for authors
        pre_authors = set()
        during_authors = set()
        
        # Load pre-COVID comments
        if os.path.exists(pre_comments_path):
            try:
                df_pre_comments = pd.read_parquet(pre_comments_path, columns=['author'])
                pre_authors.update(df_pre_comments['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {pre_comments_path}: {e}")
        
        # Load pre-COVID submissions
        if os.path.exists(pre_submissions_path):
            try:
                df_pre_submissions = pd.read_parquet(pre_submissions_path, columns=['author'])
                pre_authors.update(df_pre_submissions['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {pre_submissions_path}: {e}")
        
        # Load during-COVID comments
        if os.path.exists(during_comments_path):
            try:
                df_during_comments = pd.read_parquet(during_comments_path, columns=['author'])
                during_authors.update(df_during_comments['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {during_comments_path}: {e}")
        
        # Load during-COVID submissions
        if os.path.exists(during_submissions_path):
            try:
                df_during_submissions = pd.read_parquet(during_submissions_path, columns=['author'])
                during_authors.update(df_during_submissions['author'].dropna().unique())
            except Exception as e:
                print(f"Error reading {during_submissions_path}: {e}")
        
        # Compute new and returning users
        new_users = during_authors - pre_authors
        returning_users = during_authors.intersection(pre_authors)
        
        # Select example authors
        example_new_user: Optional[str] = next(iter(new_users), None) if new_users else None
        example_returning_user: Optional[str] = next(iter(returning_users), None) if returning_users else None
        
        # Append results
        results.append({
            'City': city,
            'New_Users': len(new_users),
            'Returning_Users': len(returning_users),
            'Example_New_User': example_new_user,
            'Example_Returning_User': example_returning_user
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to Parquet if output_path is specified
    if output_path:
        try:
            results_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
            print(f"User statistics saved to {output_path}")
        except Exception as e:
            print(f"Error saving to Parquet file: {e}")
    
    if return_df:
        return results_df
    else:
        # Plotting the data
        try:
            plt.figure(figsize=(16, 10))  # Increased figure size for better readability
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
            plt.xticks(indices, results_df['City'], rotation=90, ha='center')  # Rotated labels to 90 degrees
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.title('New and Returning Users per City Subreddit', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Added horizontal grid lines
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error during plotting: {e}")
        return None


# def plot_comments_distribution(
#     parquet_file_path,
#     title='Comments Distribution',
#     xlabel='Number of Comments',
#     ylabel='Number of Posts',
#     log_x=False,
#     log_y=False,
#     figsize=(10, 6),
# ):
#     """
#     Plots the distribution of the number of comments on posts.

#     Parameters:
#     - parquet_file_path (str): Path to the Parquet file containing 'num_comments' field.
#     - log_x (bool): If True, set the x-axis to logarithmic scale.
#     - log_y (bool): If True, set the y-axis to logarithmic scale.
#     - title (str): Title of the plot.
#     - xlabel (str): Label for the x-axis.
#     - ylabel (str): Label for the y-axis.
#     - figsize (tuple): Size of the plot figure.

#     Returns:
#     - None
#     """

#     # Read the Parquet file
#     try:
#         df = pd.read_parquet(parquet_file_path)
#     except Exception as e:
#         print(f"Error reading the Parquet file: {e}")
#         return

#     if 'num_comments' not in df.columns:
#         print("The Parquet file does not contain a 'num_comments' column.")
#         return

#     # Handle missing or invalid values
#     df = df.dropna(subset=['num_comments'])
#     df = df[df['num_comments'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]

#     # Convert num_comments to integer if necessary
#     df['num_comments'] = df['num_comments'].astype(int)

#     # Calculate the distribution
#     comments_counts = df['num_comments'].value_counts().sort_index()

#     # Prepare the plot
#     plt.figure(figsize=figsize)
#     sns.set(style="whitegrid")

#     # Choose plot type based on the range of data
#     # If the data is sparse, a scatter plot might be more appropriate
#     if comments_counts.max() > 1000:
#         # For large ranges, use a scatter plot
#         plt.scatter(comments_counts.index, comments_counts.values, alpha=0.6, edgecolor='b')
#     else:
#         # For smaller ranges, use a bar plot
#         plt.bar(comments_counts.index, comments_counts.values, color='skyblue')

#     # Set logarithmic scales if specified
#     if log_x:
#         plt.xscale('log')
#     if log_y:
#         plt.yscale('log')

#     # Set labels and title
#     plt.xlabel(xlabel, fontsize=12)
#     plt.ylabel(ylabel, fontsize=12)
#     plt.title(title, fontsize=14)

#     # Improve layout
#     plt.tight_layout()

#     # Show the plot
#     plt.show()


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

    # Read the Parquet file
    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"Error reading the Parquet file: {e}")
        return

    if 'num_comments' not in df.columns:
        print("The Parquet file does not contain a 'num_comments' column.")
        return

    # Handle missing or invalid values
    df = df.dropna(subset=['num_comments'])
    df = df[df['num_comments'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]

    # Convert num_comments to integer if necessary
    df['num_comments'] = df['num_comments'].astype(int)

    # Calculate the distribution
    comments_counts = df['num_comments'].value_counts().sort_index()

    # Prepare the plot
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")

    plt.scatter(comments_counts.index, comments_counts.values, alpha=0.6, edgecolor='b')

    # # Choose plot type based on the range of data
    # if comments_counts.max() > 1000:
    #     # For large ranges, use a scatter plot
    #     plt.scatter(comments_counts.index, comments_counts.values, alpha=0.6, edgecolor='b')
    # else:
    #     # For smaller ranges, use a bar plot
    #     plt.bar(comments_counts.index, comments_counts.values, color='skyblue')

    # Set logarithmic scales if specified
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    # Set labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # Improve layout
    plt.tight_layout()

    # Save the plot if requested
    if save_plot:
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
        plt.close()


def plot_comments_distribution_plotly(
    parquet_file_path,
    title='Comments Distribution',
    xlabel='Number of Comments',
    ylabel='Number of Posts',
    log_x=False,
    log_y=False,
):
    """
    Plots the distribution of the number of comments on posts using Plotly.

    Parameters:
    - parquet_file_path (str): Path to the Parquet file containing 'num_comments' field.
    - log_x (bool): If True, set the x-axis to logarithmic scale.
    - log_y (bool): If True, set the y-axis to logarithmic scale.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None
    """

    # Read the Parquet file
    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"Error reading the Parquet file: {e}")
        return

    if 'num_comments' not in df.columns:
        print("The Parquet file does not contain a 'num_comments' column.")
        return

    # Handle missing or invalid values
    df = df.dropna(subset=['num_comments'])
    df = df[df['num_comments'].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]

    # Convert num_comments to integer if necessary
    df['num_comments'] = df['num_comments'].astype(int)

    # Calculate the distribution
    comments_counts = df['num_comments'].value_counts().reset_index()
    comments_counts.columns = ['num_comments', 'num_posts']
    comments_counts = comments_counts.sort_values('num_comments')

    # Create the plot
    fig = px.scatter(
        comments_counts,
        x='num_comments',
        y='num_posts',
        log_x=log_x,
        log_y=log_y,
        title=title,
        labels={
            'num_comments': xlabel,
            'num_posts': ylabel
        },
        hover_data=['num_comments', 'num_posts']
    )

    fig.update_traces(marker=dict(size=8, color='blue', opacity=0.6))
    fig.update_layout(template='plotly_white')

    fig.show()


# def plot_post_lifespan(
#     posts_parquet_path,
#     comments_parquet_path,
#     units='hours',
#     title='Post Lifespan Distribution',
#     xlabel=None,
#     ylabel='Number of Posts',
#     return_data=False,
#     log_scale=(False, False),
#     verbose=False,
#     plot_type='histogram',
#     figsize=(10, 6),
# ):
#     """
#     Calculates and plots the lifespan of posts based on comment activity,
#     excluding posts with a lifespan of 0 (i.e., posts with only one comment).

#     Parameters:
#     - posts_parquet_path (str): Path to the posts Parquet file containing an 'id' field.
#     - comments_parquet_path (str): Path to the comments Parquet file containing 'parent_id' and 'created_utc' fields.
#     - units (str): Time units for lifespan. Options: 'seconds', 'minutes', 'hours', 'days'. Default is 'hours'.
#     - title (str): Title of the plot. Default is 'Post Lifespan Distribution'.
#     - xlabel (str): Label for the x-axis. If None, it will be set based on the 'units'.
#     - ylabel (str): Label for the y-axis. Default is 'Number of Posts'.
#     - figsize (tuple): Size of the plot figure. Default is (10, 6).
#     - return_data (bool): If True, returns a DataFrame with post IDs and their lifespans. Default is False.
#     - plot_type (str): Type of plot to generate. Options: 'histogram', 'density', 'cdf', 'box'. Default is 'histogram'.
#     - log_scale (tuple): Tuple indicating whether to apply logarithmic scaling to the x and y axes.
#                          Example: (True, False) applies log scale to x-axis only.
#     - verbose (bool): If True, prints detailed logs for debugging purposes. Default is False.

#     Returns:
#     - If return_data=True, returns a pandas DataFrame with columns ['post_id', 'lifespan'].
#     - Otherwise, returns None.
#     """

#     # Validate 'units' parameter
#     valid_units = ['seconds', 'minutes', 'hours', 'days']
#     if units not in valid_units:
#         raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")

#     # Validate 'plot_type' parameter
#     valid_plot_types = ['histogram', 'density', 'cdf', 'box']
#     if plot_type not in valid_plot_types:
#         raise ValueError(f"Invalid plot_type '{plot_type}'. Choose from {valid_plot_types}.")

#     # Validate 'log_scale' parameter
#     if not (isinstance(log_scale, tuple) and len(log_scale) == 2 and 
#             all(isinstance(x, bool) for x in log_scale)):
#         raise ValueError("log_scale must be a tuple of two boolean values, e.g., (True, False).")

#     # Set default xlabel if not provided
#     if xlabel is None:
#         xlabel = f'Lifespan ({units})'

#     # Step 1: Read the Posts Parquet File
#     try:
#         if verbose:
#             print(f"Reading posts data from {posts_parquet_path}...")
#         posts_df = pd.read_parquet(posts_parquet_path, columns=['id'])
#     except Exception as e:
#         print(f"Error reading the posts Parquet file: {e}")
#         return

#     if 'id' not in posts_df.columns:
#         print("The posts Parquet file does not contain an 'id' column.")
#         return

#     if verbose:
#         print(f"Number of posts loaded: {len(posts_df)}")

#     # Step 2: Read the Comments Parquet File
#     try:
#         if verbose:
#             print(f"Reading comments data from {comments_parquet_path}...")
#         comments_df = pd.read_parquet(comments_parquet_path, columns=['parent_id', 'created_utc'])
#     except Exception as e:
#         print(f"Error reading the comments Parquet file: {e}")
#         return

#     required_columns = ['parent_id', 'created_utc']
#     for col in required_columns:
#         if col not in comments_df.columns:
#             print(f"The comments Parquet file does not contain a '{col}' column.")
#             return

#     if verbose:
#         print(f"Number of comments loaded: {len(comments_df)}")

#     # Step 3: Clean 'parent_id' by removing prefixes ('t1_', 't3_', etc.)
#     # Assuming 't1_' prefixes indicate comments and 't3_' indicate posts; we need to match comments to their parent posts
#     comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)

#     if verbose:
#         print("Sample of cleaned 'parent_id':")
#         print(comments_df['parent_id_clean'].head())

#     # Step 4: Ensure 'created_utc' is numeric
#     if not np.issubdtype(comments_df['created_utc'].dtype, np.number):
#         if verbose:
#             print("'created_utc' is not numeric. Attempting to convert...")
#         try:
#             comments_df['created_utc'] = pd.to_numeric(comments_df['created_utc'], errors='coerce')
#         except Exception as e:
#             print(f"Error converting 'created_utc' to numeric: {e}")
#             return

#     # Drop rows with invalid 'created_utc'
#     initial_comments = len(comments_df)
#     comments_df = comments_df.dropna(subset=['created_utc'])
#     if verbose:
#         print(f"Dropped {initial_comments - len(comments_df)} comments due to invalid 'created_utc'.")

#     # Ensure 'created_utc' is in seconds since epoch (float or int)
#     # If 'created_utc' is a datetime, convert to timestamp
#     if pd.api.types.is_datetime64_any_dtype(comments_df['created_utc']):
#         if verbose:
#             print("'created_utc' is datetime. Converting to Unix timestamp...")
#         comments_df['created_utc'] = comments_df['created_utc'].astype(np.int64) // 10**9

#     # Step 5: Merge Comments with Posts on 'parent_id_clean' == 'id'
#     if verbose:
#         print("Merging comments with posts...")
#     merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner')

#     if verbose:
#         print(f"Number of comments after merging with posts: {len(merged_df)}")

#     # Step 6: Group by 'id' and calculate min and max 'created_utc'
#     if verbose:
#         print("Calculating min and max 'created_utc' for each post...")
#     lifespan_df = merged_df.groupby('id')['created_utc'].agg(['min', 'max']).reset_index()
#     lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

#     if verbose:
#         print(f"Number of posts with comments: {len(lifespan_df)}")
#         print("Sample of lifespan data:")
#         print(lifespan_df.head())

#     # Step 7: Calculate Lifespan in Seconds
#     lifespan_df['lifespan_seconds'] = lifespan_df['max'] - lifespan_df['min']

#     # Handle negative or zero lifespans (if any)
#     invalid_lifespans = lifespan_df[lifespan_df['lifespan_seconds'] < 0]
#     if not invalid_lifespans.empty:
#         if verbose:
#             print(f"Found {len(invalid_lifespans)} posts with negative lifespans. Excluding them.")
#         lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

#     # Step 8: Convert Lifespan to Desired Units
#     if units == 'seconds':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds']
#     elif units == 'minutes':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 60
#     elif units == 'hours':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 3600
#     elif units == 'days':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 86400

#     if verbose:
#         print(f"Converted lifespans to {units}.")
#         print("Sample of converted lifespans:")
#         print(lifespan_df['lifespan'].head())

#     # Step 9: Exclude Posts with Lifespan of 0 (i.e., posts with only one comment)
#     initial_count = len(lifespan_df)
#     lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
#     filtered_count = len(lifespan_df)
#     excluded_posts = initial_count - filtered_count
#     if verbose:
#         print(f"Excluded {excluded_posts} posts with a lifespan of 0.")

#     # Check if any data remains after filtering
#     if lifespan_df.empty:
#         print("No posts with lifespan greater than 0. Cannot plot histogram.")
#         return

#     # Step 10: Plot the Lifespan Distribution
#     plt.figure(figsize=figsize)
#     sns.set(style="whitegrid")

#     # Define bins dynamically based on units
#     # Using logarithmic bins if log_scale for x is True
#     if log_scale[0]:
#         if lifespan_df['lifespan'].min() <= 0:
#             # Log scale cannot handle zero or negative values; add a small constant
#             lifespan_positive = lifespan_df['lifespan'] + 1e-6
#             bins = np.logspace(np.log10(lifespan_positive.min()), np.log10(lifespan_positive.max()), 50)
#         else:
#             bins = np.logspace(np.log10(lifespan_df['lifespan'].min()), np.log10(lifespan_df['lifespan'].max()), 50)
#     else:
#         bins = 50  # Default number of bins

#     if verbose:
#         print(f"Plotting {plot_type} with {'logarithmic' if log_scale[0] else 'linear'} x-axis.")

#     # Plot based on plot_type
#     if plot_type == 'histogram':
#         sns.histplot(
#             lifespan_df['lifespan'],
#             bins=bins,
#             kde=False,
#             color='skyblue',
#             edgecolor='black'
#         )
#     elif plot_type == 'density':
#         sns.kdeplot(
#             lifespan_df['lifespan'],
#             shade=True,
#             color='skyblue'
#         )
#     elif plot_type == 'cdf':
#         sns.ecdfplot(
#             lifespan_df['lifespan'],
#             color='skyblue'
#         )
#     elif plot_type == 'box':
#         sns.boxplot(
#             x=lifespan_df['lifespan'],
#             color='skyblue'
#         )
#     else:
#         raise ValueError(f"Unsupported plot_type '{plot_type}'.")

#     # Set logarithmic scales if specified
#     if log_scale[0]:
#         plt.xscale('log')
#     if log_scale[1]:
#         plt.yscale('log')

#     # Set labels and title
#     plt.title(title, fontsize=14)
#     plt.xlabel(xlabel, fontsize=12)
#     plt.ylabel(ylabel, fontsize=12)

#     plt.tight_layout()
#     plt.show()

#     # Step 11: Return Lifespan Data if Requested
#     if return_data:
#         return lifespan_df[['post_id', 'lifespan']]


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

    # Validate 'units' parameter
    valid_units = ['seconds', 'minutes', 'hours', 'days']
    if units not in valid_units:
        raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")

    # Validate 'plot_type' parameter
    valid_plot_types = ['histogram', 'density', 'cdf', 'box']
    if plot_type not in valid_plot_types:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Choose from {valid_plot_types}.")

    # Validate 'log_scale' parameter
    if not (isinstance(log_scale, tuple) and len(log_scale) == 2 and 
            all(isinstance(x, bool) for x in log_scale)):
        raise ValueError("log_scale must be a tuple of two boolean values, e.g., (True, False).")

    # Set default xlabel if not provided
    if xlabel is None:
        xlabel = f'Lifespan ({units})'

    # Step 1: Read the Posts Parquet File
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

    # Step 2: Read the Comments Parquet File
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

    # Step 3: Clean 'parent_id' by removing prefixes ('t1_', 't3_', etc.)
    # Assuming 't1_' prefixes indicate comments and 't3_' indicate posts; we need to match comments to their parent posts
    comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)

    if verbose:
        print("Sample of cleaned 'parent_id':")
        print(comments_df['parent_id_clean'].head())

    # Step 4: Ensure 'created_utc' is numeric
    if not np.issubdtype(comments_df['created_utc'].dtype, np.number):
        if verbose:
            print("'created_utc' is not numeric. Attempting to convert...")
        try:
            comments_df['created_utc'] = pd.to_numeric(comments_df['created_utc'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'created_utc' to numeric: {e}")
            return

    # Drop rows with invalid 'created_utc'
    initial_comments = len(comments_df)
    comments_df = comments_df.dropna(subset=['created_utc'])
    if verbose:
        print(f"Dropped {initial_comments - len(comments_df)} comments due to invalid 'created_utc'.")

    # Ensure 'created_utc' is in seconds since epoch (float or int)
    # If 'created_utc' is a datetime, convert to timestamp
    if pd.api.types.is_datetime64_any_dtype(comments_df['created_utc']):
        if verbose:
            print("'created_utc' is datetime. Converting to Unix timestamp...")
        comments_df['created_utc'] = comments_df['created_utc'].astype(np.int64) // 10**9

    # Step 5: Merge Comments with Posts on 'parent_id_clean' == 'id'
    if verbose:
        print("Merging comments with posts...")
    merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner')

    if verbose:
        print(f"Number of comments after merging with posts: {len(merged_df)}")

    # Step 6: Group by 'id' and calculate min and max 'created_utc'
    if verbose:
        print("Calculating min and max 'created_utc' for each post...")
    lifespan_df = merged_df.groupby('id')['created_utc'].agg(['min', 'max']).reset_index()
    lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

    if verbose:
        print(f"Number of posts with comments: {len(lifespan_df)}")
        print("Sample of lifespan data:")
        print(lifespan_df.head())

    # Step 7: Calculate Lifespan in Seconds
    lifespan_df['lifespan_seconds'] = lifespan_df['max'] - lifespan_df['min']

    # Handle negative or zero lifespans (if any)
    invalid_lifespans = lifespan_df[lifespan_df['lifespan_seconds'] < 0]
    if not invalid_lifespans.empty:
        if verbose:
            print(f"Found {len(invalid_lifespans)} posts with negative lifespans. Excluding them.")
        lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

    # Step 8: Convert Lifespan to Desired Units
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

    # Step 9: Exclude Posts with Lifespan of 0 (i.e., posts with only one comment)
    initial_count = len(lifespan_df)
    lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
    filtered_count = len(lifespan_df)
    excluded_posts = initial_count - filtered_count
    if verbose:
        print(f"Excluded {excluded_posts} posts with a lifespan of 0.")

    # Check if any data remains after filtering
    if lifespan_df.empty:
        print("No posts with lifespan greater than 0. Cannot plot histogram.")
        return

    # Step 10: Plot the Lifespan Distribution
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")

    # Define bins dynamically based on units.
    # Use logarithmic bins if log_scale for x is True.
    if log_scale[0]:
        if lifespan_df['lifespan'].min() <= 0:
            # Log scale cannot handle zero or negative values; add a small constant
            lifespan_positive = lifespan_df['lifespan'] + 1e-6
            bins = np.logspace(np.log10(lifespan_positive.min()), np.log10(lifespan_positive.max()), 50)
        else:
            bins = np.logspace(np.log10(lifespan_df['lifespan'].min()), np.log10(lifespan_df['lifespan'].max()), 50)
    else:
        bins = 50  # Default number of bins

    if verbose:
        print(f"Plotting {plot_type} with {'logarithmic' if log_scale[0] else 'linear'} x-axis.")

    # Plot based on plot_type
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

    # Set logarithmic scales if specified
    if log_scale[0]:
        plt.xscale('log')
    if log_scale[1]:
        plt.yscale('log')

    # Set labels and title
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.tight_layout()

    # Save the plot if requested
    if save_plot:
        # Ensure the directory exists
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        if verbose:
            print(f"Plot saved to {plot_path}")

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
        plt.close()

    # Step 11: Return Lifespan Data if Requested
    if return_data:
        return lifespan_df[['post_id', 'lifespan']]


# def plot_post_lifespan_timeseries(
#     posts_parquet_path,
#     comments_parquet_path,
#     aggregation='monthly',
#     units='hours',
#     statistic='mean',
#     title='Post Lifespan Over Time',
#     xlabel=None,
#     ylabel=None,
#     figsize=(14, 7),
#     return_data=False,
#     log_scale=False,
#     verbose=False
# ):
#     """
#     Calculates and plots the lifespan of posts over time, aggregated by a specified interval.
    
#     Parameters:
#     - posts_parquet_path (str): Path to the posts Parquet file containing 'id' and 'created_utc' fields.
#     - comments_parquet_path (str): Path to the comments Parquet file containing 'parent_id' and 'created_utc' fields.
#     - aggregation (str): Time interval for aggregation. Options: 'daily', 'weekly', 'monthly', 'yearly'. Default is 'monthly'.
#     - units (str): Time units for lifespan. Options: 'seconds', 'minutes', 'hours', 'days'. Default is 'hours'.
#     - statistic (str): Statistic to plot. Options: 'mean', 'median', 'count'. Default is 'mean'.
#     - title (str): Title of the plot. Default is 'Post Lifespan Over Time'.
#     - xlabel (str): Label for the x-axis. If None, it will be set based on aggregation.
#     - ylabel (str): Label for the y-axis. If None, it will be set based on statistic and units.
#     - figsize (tuple): Size of the plot figure. Default is (14, 7).
#     - return_data (bool): If True, returns a DataFrame with aggregated lifespan statistics. Default is False.
#     - log_scale (bool): If True, applies logarithmic scaling to the y-axis. Default is False.
#     - verbose (bool): If True, prints detailed logs for debugging purposes. Default is False.
    
#     Returns:
#     - If return_data=True, returns a pandas DataFrame with aggregated statistics.
#     - Otherwise, returns None.
#     """
    
#     # Validate 'aggregation' parameter
#     valid_aggregations = ['daily', 'weekly', 'monthly', 'yearly']
#     if aggregation not in valid_aggregations:
#         raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from {valid_aggregations}.")
    
#     # Validate 'units' parameter
#     valid_units = ['seconds', 'minutes', 'hours', 'days']
#     if units not in valid_units:
#         raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")
    
#     # Validate 'statistic' parameter
#     valid_statistics = ['mean', 'median', 'count']
#     if statistic not in valid_statistics:
#         raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")
    
#     # Set default labels if not provided
#     if xlabel is None:
#         xlabel = f'Time ({aggregation.capitalize()})'
#     if ylabel is None:
#         if statistic in ['mean', 'median']:
#             ylabel = f'Post Lifespan ({units.capitalize()})'
#         elif statistic == 'count':
#             ylabel = 'Number of Posts'
    
#     if verbose:
#         print("Starting plot_post_lifespan_timeseries function...")
#         print(f"Aggregation: {aggregation}, Units: {units}, Statistic: {statistic}")
    
#     # Step 1: Read the Posts Parquet File
#     try:
#         if verbose:
#             print(f"Reading posts data from {posts_parquet_path}...")
#         posts_df = pd.read_parquet(posts_parquet_path, columns=['id', 'created_utc'])
#     except Exception as e:
#         print(f"Error reading the posts Parquet file: {e}")
#         return
    
#     if 'id' not in posts_df.columns or 'created_utc' not in posts_df.columns:
#         print("The posts Parquet file must contain 'id' and 'created_utc' columns.")
#         return
    
#     if verbose:
#         print(f"Number of posts loaded: {len(posts_df)}")
    
#     # Convert 'created_utc' to datetime
#     if not np.issubdtype(posts_df['created_utc'].dtype, np.number):
#         try:
#             posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
#             if verbose:
#                 print("'created_utc' in posts converted to datetime.")
#         except Exception as e:
#             print(f"Error converting 'created_utc' in posts to datetime: {e}")
#             return
#     else:
#         # Assuming 'created_utc' is in seconds since epoch
#         posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
#         if verbose:
#             print("'created_utc' in posts converted from Unix timestamp to datetime.")
    
#     # Step 2: Read the Comments Parquet File
#     try:
#         if verbose:
#             print(f"Reading comments data from {comments_parquet_path}...")
#         comments_df = pd.read_parquet(comments_parquet_path, columns=['parent_id', 'created_utc'])
#     except Exception as e:
#         print(f"Error reading the comments Parquet file: {e}")
#         return
    
#     required_columns = ['parent_id', 'created_utc']
#     for col in required_columns:
#         if col not in comments_df.columns:
#             print(f"The comments Parquet file does not contain a '{col}' column.")
#             return
    
#     if verbose:
#         print(f"Number of comments loaded: {len(comments_df)}")
    
#     # Step 3: Clean 'parent_id' by removing prefixes ('t1_', 't3_', etc.)
#     comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)
    
#     if verbose:
#         print("Sample of cleaned 'parent_id':")
#         print(comments_df['parent_id_clean'].head())
    
#     # Step 4: Ensure 'created_utc' is in datetime
#     if not np.issubdtype(comments_df['created_utc'].dtype, np.number):
#         try:
#             comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
#             if verbose:
#                 print("'created_utc' in comments converted to datetime.")
#         except Exception as e:
#             print(f"Error converting 'created_utc' in comments to datetime: {e}")
#             return
#     else:
#         # Assuming 'created_utc' is in seconds since epoch
#         comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
#         if verbose:
#             print("'created_utc' in comments converted from Unix timestamp to datetime.")
    
#     # Step 5: Merge Comments with Posts on 'parent_id_clean' == 'id'
#     if verbose:
#         print("Merging comments with posts...")
#     merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner', suffixes=('_comment', '_post'))
    
#     if verbose:
#         print(f"Number of comments after merging with posts: {len(merged_df)}")
#         print("Sample of merged data:")
#         print(merged_df.head())
    
#     # Step 6: Group by 'id_post' and calculate min and max 'created_utc_comment'
#     if verbose:
#         print("Calculating min and max 'created_utc' for each post...")
#     lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
#     lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

#     if verbose:
#         print(f"Number of posts with comments: {len(lifespan_df)}")
#         print("Sample of lifespan data:")
#         print(lifespan_df.head())
    
#     # Step 7: Calculate Lifespan in Seconds
#     lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()
    
#     # Handle negative or zero lifespans (if any)
#     invalid_lifespans = lifespan_df[lifespan_df['lifespan_seconds'] < 0]
#     if not invalid_lifespans.empty:
#         if verbose:
#             print(f"Found {len(invalid_lifespans)} posts with negative lifespans. Excluding them.")
#         lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]
    
#     # Step 8: Convert Lifespan to Desired Units
#     if units == 'seconds':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds']
#     elif units == 'minutes':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 60
#     elif units == 'hours':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 3600
#     elif units == 'days':
#         lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 86400
    
#     if verbose:
#         print(f"Converted lifespans to {units}.")
#         print("Sample of converted lifespans:")
#         print(lifespan_df['lifespan'].head())
    
#     # Step 9: Exclude Posts with Lifespan of 0 (i.e., posts with only one comment)
#     initial_count = len(lifespan_df)
#     lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
#     filtered_count = len(lifespan_df)
#     excluded_posts = initial_count - filtered_count
#     if verbose:
#         print(f"Excluded {excluded_posts} posts with a lifespan of 0.")
    
#     # Check if any data remains after filtering
#     if lifespan_df.empty:
#         print("No posts with lifespan greater than 0. Cannot plot timeseries.")
#         return
    
#     # Step 10: Merge Lifespan with Post Creation Time
#     if verbose:
#         print("Merging lifespan data with post creation times...")
#     lifespan_df = lifespan_df.merge(posts_df[['id', 'created_utc']], left_on='post_id', right_on='id', how='left')
#     lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
#     lifespan_df.drop('id', axis=1, inplace=True)
    
#     # Step 11: Aggregate Lifespan Statistics Over Time
#     if verbose:
#         print(f"Aggregating lifespans by {aggregation}...")
#     # Set post creation time as index
#     lifespan_df.set_index('post_created_time', inplace=True)
    
#     # Resample and aggregate
#     if aggregation == 'daily':
#         resample_rule = 'D'
#     elif aggregation == 'weekly':
#         resample_rule = 'W'
#     elif aggregation == 'monthly':
#         resample_rule = 'M'
#     elif aggregation == 'yearly':
#         resample_rule = 'Y'
    
#     if statistic in ['mean', 'median']:
#         aggregated = lifespan_df['lifespan'].resample(resample_rule).agg(statistic)
#     elif statistic == 'count':
#         aggregated = lifespan_df['lifespan'].resample(resample_rule).count()
    
#     if verbose:
#         print("Aggregated lifespan statistics:")
#         print(aggregated.head())
    
#     # Step 12: Plot the Timeseries
#     plt.figure(figsize=figsize)
#     sns.set(style="whitegrid")
    
#     if statistic in ['mean', 'median']:
#         sns.lineplot(x=aggregated.index, y=aggregated.values, marker='o', color='skyblue')
#     elif statistic == 'count':
#         sns.barplot(x=aggregated.index, y=aggregated.values, color='skyblue')
    
#     # Apply log scale if specified
#     if log_scale:
#         plt.yscale('log')
    
#     # Set labels and title
#     plt.title(title, fontsize=16)
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(ylabel, fontsize=14)
    
#     # Improve date formatting on x-axis
#     if aggregation in ['monthly', 'yearly']:
#         plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Step 13: Return Aggregated Data if Requested
#     if return_data:
#         aggregated_df = aggregated.reset_index()
#         aggregated_df.columns = ['Time', f'Lifespan_{statistic}_{units}']
#         return aggregated_df


def plot_post_count(
    posts_file,
    comments_file,          # Not used in this version; kept for interface consistency.
    precovid_posts_file,
    precovid_comments_file, # Not used in this version; kept for interface consistency.
    city='',
    resample_unit='W',  # e.g., 'W' for weekly, 'D' for daily, 'M' for monthly
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
        # Load posts data
        print(f"Loading posts data from {posts_file}...")
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")

        # Convert Unix timestamps to datetime
        print("Converting timestamps...")
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')

        # Sort posts by creation time
        posts.sort_values('created_datetime', inplace=True)

        # Set the creation datetime as the index for resampling
        posts.set_index('created_datetime', inplace=True)

        # Resample by creation time to count the number of posts per period
        print(f"Resampling by '{resample_unit}' to compute the number of posts...")
        post_counts = posts.resample(resample_unit).size()
        return post_counts

    # Process pre-COVID posts data and restrict to a desired period (e.g., Oct-Dec 2019)
    print("Processing pre-COVID posts data...")
    precovid_counts = process_data(precovid_posts_file)
    precovid_counts = precovid_counts['2019-10-01':'2019-12-31']  # Adjust as needed

    # Process main (COVID) posts data
    print("Processing main posts data...")
    main_counts = process_data(posts_file)

    # Combine the two time series into one
    print("Combining datasets...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    # Plotting the time series
    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', label='Number of Posts', color='blue')

    # Mark the COVID start date (January 2020)
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
    resample_unit='W',  # e.g., 'W' for weekly, 'D' for daily, 'M' for monthly
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
        # Load comments data
        print(f"Loading comments data from {comments_file}...")
        comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")

        # Convert timestamps to datetime
        print("Converting timestamps...")
        comments['created_datetime'] = pd.to_datetime(comments['created_utc'], unit='s')

        # Sort by creation time
        comments.sort_values('created_datetime', inplace=True)

        # Set the creation datetime as the index for resampling
        comments.set_index('created_datetime', inplace=True)

        # Resample by comment creation time to count the number of comments per period
        print(f"Resampling by '{resample_unit}' to compute the number of comments...")
        comment_counts = comments.resample(resample_unit).size()
        return comment_counts

    # Process pre-COVID comments data (e.g., restricting to Oct-Dec 2019)
    print("Processing pre-COVID comments data...")
    precovid_counts = process_data(precovid_comments_file)
    precovid_counts = precovid_counts['2019-10-01':'2019-12-31']  # Adjust this window as needed

    # Process main (COVID-era) comments data
    print("Processing main comments data...")
    main_counts = process_data(comments_file)

    # Combine the two time series into one
    print("Combining datasets...")
    combined_counts = pd.concat([precovid_counts, main_counts])

    # Plotting the time series
    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_counts.plot(marker='o', linestyle='-', label='Number of Comments', color='blue')

    # Mark the COVID start date (January 2020)
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
    
    # Validate 'aggregation' parameter
    valid_aggregations = ['daily', 'weekly', 'monthly', 'yearly']
    if aggregation not in valid_aggregations:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from {valid_aggregations}.")
    
    # Validate 'units' parameter
    valid_units = ['seconds', 'minutes', 'hours', 'days']
    if units not in valid_units:
        raise ValueError(f"Invalid units '{units}'. Choose from {valid_units}.")
    
    # Validate 'statistic' parameter
    valid_statistics = ['mean', 'median', 'count']
    if statistic not in valid_statistics:
        raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")
    
    # Set default labels if not provided
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
    
    # Step 1: Read the Posts Parquet File
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
    
    # Convert 'created_utc' to datetime
    if not np.issubdtype(posts_df['created_utc'].dtype, np.number):
        try:
            posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
            if verbose:
                print("'created_utc' in posts converted to datetime.")
        except Exception as e:
            print(f"Error converting 'created_utc' in posts to datetime: {e}")
            return
    else:
        # Assuming 'created_utc' is in seconds since epoch
        posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')
        if verbose:
            print("'created_utc' in posts converted from Unix timestamp to datetime.")
    
    # Step 2: Read the Comments Parquet File
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
    
    # Step 3: Clean 'parent_id' by removing prefixes ('t1_', 't3_', etc.)
    comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)
    
    if verbose:
        print("Sample of cleaned 'parent_id':")
        print(comments_df['parent_id_clean'].head())
    
    # Step 4: Ensure 'created_utc' is in datetime for comments
    if not np.issubdtype(comments_df['created_utc'].dtype, np.number):
        try:
            comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
            if verbose:
                print("'created_utc' in comments converted to datetime.")
        except Exception as e:
            print(f"Error converting 'created_utc' in comments to datetime: {e}")
            return
    else:
        # Assuming 'created_utc' is in seconds since epoch
        comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')
        if verbose:
            print("'created_utc' in comments converted from Unix timestamp to datetime.")
    
    # Step 5: Merge Comments with Posts on 'parent_id_clean' == 'id'
    if verbose:
        print("Merging comments with posts...")
    merged_df = comments_df.merge(posts_df, left_on='parent_id_clean', right_on='id', how='inner', suffixes=('_comment', '_post'))
    
    if verbose:
        print(f"Number of comments after merging with posts: {len(merged_df)}")
        print("Sample of merged data:")
        print(merged_df.head())
    
    # Step 6: Group by 'id' and calculate min and max 'created_utc_comment'
    if verbose:
        print("Calculating min and max 'created_utc' for each post...")
    lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
    lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

    if verbose:
        print(f"Number of posts with comments: {len(lifespan_df)}")
        print("Sample of lifespan data:")
        print(lifespan_df.head())
    
    # Step 7: Calculate Lifespan in Seconds
    lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()
    
    # Handle negative or zero lifespans (if any)
    invalid_lifespans = lifespan_df[lifespan_df['lifespan_seconds'] < 0]
    if not invalid_lifespans.empty:
        if verbose:
            print(f"Found {len(invalid_lifespans)} posts with negative lifespans. Excluding them.")
        lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]
    
    # Step 8: Convert Lifespan to Desired Units
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
    
    # Step 9: Exclude Posts with Lifespan of 0 (i.e., posts with only one comment)
    initial_count = len(lifespan_df)
    lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
    filtered_count = len(lifespan_df)
    excluded_posts = initial_count - filtered_count
    if verbose:
        print(f"Excluded {excluded_posts} posts with a lifespan of 0.")
    
    # Check if any data remains after filtering
    if lifespan_df.empty:
        print("No posts with lifespan greater than 0. Cannot plot timeseries.")
        return
    
    # Step 10: Merge Lifespan with Post Creation Time
    if verbose:
        print("Merging lifespan data with post creation times...")
    lifespan_df = lifespan_df.merge(posts_df[['id', 'created_utc']], left_on='post_id', right_on='id', how='left')
    lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
    lifespan_df.drop('id', axis=1, inplace=True)
    
    # Step 11: Aggregate Lifespan Statistics Over Time
    if verbose:
        print(f"Aggregating lifespans by {aggregation}...")
    # Set post creation time as index
    lifespan_df.set_index('post_created_time', inplace=True)
    
    # Determine the resampling rule based on aggregation
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
    
    # Step 12: Plot the Timeseries
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")
    
    if statistic in ['mean', 'median']:
        sns.lineplot(x=aggregated.index, y=aggregated.values, marker='o', color='skyblue')
    elif statistic == 'count':
        sns.barplot(x=aggregated.index, y=aggregated.values, color='skyblue')
    
    # Apply log scale if specified
    if log_scale:
        plt.yscale('log')
    
    # Set labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Improve date formatting on x-axis if required
    if aggregation in ['monthly', 'yearly']:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot if requested
    if save_plot:
        # Ensure the directory for the plot exists
        plot_dir = os.path.dirname(plot_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        if verbose:
            print(f"Plot saved to {plot_path}")
    
    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
        plt.close()
    
    # Step 13: Return Aggregated Data if Requested
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
    # Step 1: Read the posts and comments data
    print("Reading posts data...")
    posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
    print(f"Number of posts: {len(posts)}")

    print("Reading comments data...")
    comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
    print(f"Number of comments: {len(comments)}")

    # Step 2: Rename 'created_utc' columns to avoid confusion after merge
    print("Renaming 'created_utc' columns...")
    posts = posts.rename(columns={'created_utc': 'created_utc_post'})
    comments = comments.rename(columns={'created_utc': 'created_utc_comment'})

    # Step 3: Prepare the posts DataFrame
    print("Preparing posts data...")
    posts['parent_id'] = 't3_' + posts['id']  # Reddit post IDs are prefixed with 't3_'
    posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')

    # Step 4: Merge comments with posts to associate each comment with its post
    print("Merging comments with posts...")
    merged = comments.merge(
        posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
        on='parent_id',
        how='inner'  # Change to 'left' if you want to include posts without comments
    )
    print(f"Merged DataFrame size: {merged.shape}")

    # Step 5: Find the first comment for each post
    print("Identifying first comments for each post...")
    merged_sorted = merged.sort_values(['parent_id', 'created_utc_comment'])
    first_comments = merged_sorted.groupby('parent_id').first().reset_index()
    print(f"Number of posts with at least one comment: {len(first_comments)}")

    # Step 6: Calculate the response time
    print("Calculating response times...")
    first_comments['response_time_seconds'] = first_comments['created_utc_comment'] - first_comments['created_utc_post']

    # Convert response time to the desired unit
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

    # Step 7: Assign the post creation time as the time index
    first_comments['created_datetime_post'] = pd.to_datetime(first_comments['created_datetime_post'])
    first_comments.set_index('created_datetime_post', inplace=True)

    # Step 8: Resample and calculate the average response time
    print(f"Resampling data with time unit: {time_unit}")
    avg_response = first_comments['response_time'].resample(time_unit).mean()

    # Step 9: Plot the results
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

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
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

    # Step 1: Read the posts and comments data
    print("Reading posts data...")
    posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
    print(f"Number of posts: {len(posts)}")

    print("Reading comments data...")
    comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
    print(f"Number of comments: {len(comments)}")

    # Step 2: Rename 'created_utc' columns to avoid confusion after merge
    print("Renaming 'created_utc' columns...")
    posts = posts.rename(columns={'created_utc': 'created_utc_post'})
    comments = comments.rename(columns={'created_utc': 'created_utc_comment'})

    # Step 3: Prepare the posts DataFrame
    print("Preparing posts data...")
    posts['parent_id'] = 't3_' + posts['id']  # Reddit post IDs are prefixed with 't3_'
    posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')

    # Step 4: Merge comments with posts to associate each comment with its post
    print("Merging comments with posts...")
    merged = comments.merge(
        posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
        on='parent_id',
        how='inner'
    )
    print(f"Merged DataFrame size: {merged.shape}")

    # Step 5: Identify the first comment for each post
    print("Identifying first comments for each post...")
    merged_sorted = merged.sort_values(['parent_id', 'created_utc_comment'])
    first_comments = merged_sorted.groupby('parent_id').first().reset_index()
    print(f"Number of posts with at least one comment: {len(first_comments)}")

    # Step 6: Calculate the response time in seconds
    print("Calculating response times...")
    first_comments['response_time_seconds'] = (
        first_comments['created_utc_comment'] - first_comments['created_utc_post']
    )

    # Step 7: Filter out responses that occur more than 24 hours (86,400 seconds) after the post
    first_comments = first_comments[first_comments['response_time_seconds'] <= 86400]

    # Step 8: Convert response time to the desired unit and define y-axis label
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

    # Step 9: Set the post creation time as the DataFrame index
    first_comments['created_datetime_post'] = pd.to_datetime(first_comments['created_datetime_post'])
    first_comments.set_index('created_datetime_post', inplace=True)

    # Step 10: Resample the data and compute the average response time per period
    print(f"Resampling data with time unit: {time_unit}")
    avg_response = first_comments['response_time'].resample(time_unit).mean()

    # Step 11: Plot the results
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


def plot_comment_percentage(posts_file, comments_file, city='', time_unit='W', save_plot=False, plot_path='comment_percentage.png', display_plot=True):
    """
    Plots the percentage of posts that receive at least one comment, aggregated by a specified time interval (weekly or monthly).

    Parameters:
    - posts_file (str): Path to the posts Parquet file.
    - comments_file (str): Path to the comments Parquet file.
    - time_unit (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
    - save_plot (bool): Whether to save the plot to a file.
    - plot_path (str): File path to save the plot.

    Returns:
    - None: Displays and optionally saves a plot of the percentage of posts with comments.
    """

    # Step 1: Read the posts and comments data
    print("Reading posts data...")
    posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
    print(f"Number of posts: {len(posts)}")

    print("Reading comments data...")
    comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
    print(f"Number of comments: {len(comments)}")

    # Step 2: Prepare the posts DataFrame
    print("Preparing posts data...")
    posts['parent_id'] = 't3_' + posts['id']  # Reddit post IDs are prefixed with 't3_'
    posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime_post', inplace=True)

    # Step 3: Identify which posts have at least one comment
    # Extract unique parent_ids from comments
    print("Determining posts that received comments...")
    commented_posts = comments['parent_id'].unique()
    posts['has_comment'] = posts['parent_id'].isin(commented_posts)

    # Step 4: Resample posts by the specified time unit
    print(f"Resampling data with time unit: {time_unit}")
    # Calculate total number of posts and number of posts with comments for each period
    posts_per_period = posts.resample(time_unit).size()
    posts_with_comments_per_period = posts[posts['has_comment']].resample(time_unit).size()

    # Step 5: Calculate the percentage of posts that have at least one comment
    percentage_with_comments = (posts_with_comments_per_period / posts_per_period) * 100

    # Step 6: Plot the results
    print("Plotting the percentage of posts with comments...")
    plt.figure(figsize=(12, 6))
    percentage_with_comments.plot(marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Percentage of Posts with Comments (%)')
    plt.title(f'{city} Percentage of Posts Receiving Comments Over Time')
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
        plt.close()


def plot_comment_percentage_time_window(
    posts_file,
    comments_file,
    precovid_posts_file,
    precovid_comments_file,
    city='',
    resample_unit='W',  # e.g., 'W' for weekly, 'D', 'M' (for plotting)
    save_plot=False,
    plot_path='comment_percentage_7day.png',
    display_plot=True
):
    import pandas as pd
    import matplotlib.pyplot as plt

    def process_data(posts_file, comments_file):
        # Load posts and comments
        print(f"Loading posts data from {posts_file}...")
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
        print(f"Number of posts: {len(posts)}")

        print(f"Loading comments data from {comments_file}...")
        comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
        print(f"Number of comments: {len(comments)}")

        # Convert timestamps to datetime
        print("Converting timestamps...")
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc'], unit='s')

        # Attach 'parent_id' in posts
        posts['parent_id'] = 't3_' + posts['id']

        # Sort data by time
        posts.sort_values('created_datetime_post', inplace=True)
        comments.sort_values('created_datetime_comment', inplace=True)

        # Group comments by parent_id
        comment_groups = comments.groupby('parent_id')['created_datetime_comment']

        # Check if there's at least one comment in [post_time, post_time + 7 days)
        print("Checking comments in [post_time, post_time + 7 days) for each post...")

        def has_comment_in_7day_window(row):
            pid = row['parent_id']
            post_time = row['created_datetime_post']
            window_end = post_time + pd.Timedelta(days=7)

            # Get all comment times for this parent_id
            if pid not in comment_groups.groups:
                return False
            c_times = comment_groups.get_group(pid)

            # Check for comments in the time window
            in_window = c_times[(c_times >= post_time) & (c_times < window_end)]
            return len(in_window) > 0

        posts['has_comment_7day'] = posts.apply(has_comment_in_7day_window, axis=1)

        # Resample by post creation time to compute the percentage
        print(f"Resampling by '{resample_unit}' to compute the percentage of posts with 7-day comments...")
        posts.set_index('created_datetime_post', inplace=True)
        total_posts = posts.resample(resample_unit).size()
        commented_posts = posts[posts['has_comment_7day']].resample(resample_unit).size()
        percentage_with_comments = (commented_posts / total_posts) * 100

        return percentage_with_comments

    # Process pre-COVID and COVID data
    print("Processing data...")
    precovid_percentage = process_data(precovid_posts_file, precovid_comments_file)
    precovid_percentage = precovid_percentage['2019-10-01':'2019-12-31']  # Oct to Dec 2019
    main_percentage = process_data(posts_file, comments_file)

    # Combine the datasets into one time series
    print("Combining datasets...")
    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    # Plot
    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    combined_percentage.plot(marker='o', linestyle='-', label='Percentage of Posts with Comments', color='blue')
    
    # Mark the COVID start date (January 2020)
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

    # Step 0: Ensure VADER lexicon is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Step 1: Read the posts data
    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    # Step 2: Handle missing or empty text data
    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    # Step 3: Compute sentiment scores
    print("Computing sentiment scores...")
    # Define a function to compute sentiment
    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    # Apply the sentiment function to the text column
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    # Step 4: Convert 'created_utc' to datetime and set as index
    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    # Step 5: Resample and calculate average sentiment
    print(f"Resampling data with time unit: {time_unit}")
    avg_sentiment = posts['sentiment'].resample(time_unit).mean()

    # Step 6: Plot the results
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

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        # Otherwise, close the plot so it doesn't block execution
        plt.close()


def plot_negative_sentiment_count(
        posts_file,
        city='',
        text_column='selftext',
        time_unit='W',              # 'W' for weekly, 'M' for monthly, etc.
        threshold=-0.05,            # sentiment threshold below which a post is considered "negative"
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

    # Step 0: Ensure VADER lexicon is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Step 1: Read the posts data
    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    # Step 2: Handle missing or empty text data
    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    # Remove rows where the text column is just empty or whitespace
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    # Step 3: Compute sentiment scores
    print("Computing sentiment scores...")

    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    # Step 4: Convert 'created_utc' to datetime and set as index
    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    # Step 5: Identify negative posts and resample by time
    print("Identifying negative posts and resampling data...")
    # Mark posts as 1 if sentiment < threshold, else 0
    posts['is_negative'] = (posts['sentiment'] < threshold).astype(int)
    
    # Resample to count how many negative posts per time unit
    negative_post_count = posts['is_negative'].resample(time_unit).sum()

    # Step 6: Plot the results
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

    # Only display the plot if display_plot is True
    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_positive_sentiment_count(
        posts_file,
        city='',
        text_column='selftext',
        time_unit='W',              # 'W' for weekly, 'M' for monthly, etc.
        threshold=0.05,            # sentiment threshold above which a post is considered "positive"
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

    # Step 0: Ensure VADER lexicon is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Step 1: Read the posts data
    print("Reading posts data...")
    try:
        posts = pd.read_parquet(posts_file, columns=['id', 'created_utc', text_column])
    except Exception as e:
        raise ValueError(f"Error reading posts file: {e}")
    print(f"Number of posts: {len(posts)}")

    # Step 2: Handle missing or empty text data
    print("Handling missing or empty text data...")
    posts[text_column] = posts[text_column].fillna('').astype(str)
    initial_count = len(posts)
    posts = posts[posts[text_column].str.strip() != '']
    final_count = len(posts)
    print(f"Removed {initial_count - final_count} posts with empty {text_column}.")

    # Step 3: Compute sentiment scores
    print("Computing sentiment scores...")
    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']
    
    posts['sentiment'] = posts[text_column].apply(get_sentiment)

    # Step 4: Convert 'created_utc' to datetime and set as index
    print("Converting 'created_utc' to datetime...")
    posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
    posts.set_index('created_datetime', inplace=True)

    # Step 5: Identify positive posts and resample by time
    print("Identifying positive posts and resampling data...")
    # Mark posts as 1 if sentiment > threshold, else 0
    posts['is_positive'] = (posts['sentiment'] > threshold).astype(int)
    
    # Resample to count how many positive posts per time unit
    positive_post_count = posts['is_positive'].resample(time_unit).sum()

    # Step 6: Plot the results
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

    # Only display the plot if display_plot is True
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
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    for city in city_names:
        for file_type in ["comments", "submissions"]:
            file_name = f"{city}_{file_type}.parquet"
            source_path = os.path.join(source_folder, file_name)

            # Check if the file exists
            if os.path.exists(source_path):
                try:
                    # Load the parquet file into a DataFrame
                    df = pd.read_parquet(source_path)

                    # Filter out entries where the author is "AutoModerator"
                    filtered_df = df[df['author'] != "AutoModerator"]

                    # Save the filtered DataFrame to the target folder
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
    # Create the output directory if it doesn't exist
    output_dir = "../graphs"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        # Generate standardized file paths using the city key.
        # Removing spaces and converting to lowercase ensures consistency.
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        precovid_submissions_path = f"../precovid_data_parquet/{city_lower}_submissions.parquet"
        precovid_comments_path = f"../precovid_data_parquet/{city_lower}_comments.parquet"

        # # Plot and save each graph
        # # --- Plot post sentiment ---
        # plot_post_sentiment(submissions_path, city_name, text_column='selftext', time_unit='W', save_plot=True, plot_path=f'../graphs/{city_key}_post_sentiment.png', display_plot=False)
        # # sentiment_filename = os.path.join(output_dir, f"{city_key}_post_sentiment.png")
        # # plt.savefig(sentiment_filename)
        # plt.clf()  # Clear the current figure for the next plot

        # plot_negative_sentiment_count(
        #     submissions_path,
        #     city=city_name,
        #     text_column='selftext',
        #     time_unit='W',              # 'W' for weekly, 'M' for monthly, etc.
        #     threshold=-0.05,            # sentiment threshold below which a post is considered "negative"
        #     save_plot=True,
        #     plot_path=f'../graphs/{city_key}_negative_sentiment_count.png',
        #     display_plot=False
        # )
        # plt.clf()

        # plot_positive_sentiment_count(
        #     submissions_path,
        #     city=city_name,
        #     text_column='selftext',
        #     time_unit='W',              # 'W' for weekly, 'M' for monthly, etc.
        #     threshold=0.05,            # sentiment threshold above which a post is considered "positive"
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
        #     resample_unit='W',  # 'W' for weekly, 'D' for daily, 'M' for monthly, etc.
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
        #     resample_unit='W',  # 'W' for weekly, 'D' for daily, 'M' for monthly, etc.
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


# def apply_liwc_and_compute_avg_with_dict(comments_parquet, submissions_parquet, dictionary_file, category_file):
#     # Load the Parquet files into DataFrames
#     comments_df = pd.read_parquet(comments_parquet)
#     submissions_df = pd.read_parquet(submissions_parquet)
    
#     # Combine the data into one DataFrame containing all relevant self texts (comments + submissions)
#     combined_df = pd.concat([comments_df[['body']], submissions_df[['selftext']]], ignore_index=True)
    
#     # Standardize the text column name to 'text' for consistency
#     combined_df['text'] = combined_df['body'].fillna(combined_df['selftext'])  # Replace NaNs in 'body' with 'selftext' values
#     combined_df.drop(columns=['body', 'selftext'], inplace=True)  # Drop the original 'body' and 'selftext' columns after merging
    
#     # Load the LIWC categories and dictionary
#     liwc_categories = pd.read_csv(category_file, delimiter="\t", header=None, names=["Category", "Abbreviation"])
    
#     # Create a dictionary to map category codes to names
#     category_map = dict(zip(liwc_categories['Abbreviation'], liwc_categories['Category']))
    
#     # Initialize the LIWC dictionary
#     liwc_dict = {}

#     # Load and parse the LIWC dictionary file
#     with open(dictionary_file, "r") as dic_file:
#         for line in dic_file:
#             parts = line.strip().split('\t')
#             word = parts[0].lower()
#             categories = parts[1:]
#             for cat in categories:
#                 if cat and cat != '':  # Skip empty categories
#                     # Map numeric category codes to human-readable names
#                     if cat.isdigit() and int(cat) in category_map:
#                         cat_name = category_map[int(cat)]
#                     else:
#                         cat_name = cat
#                     if cat_name not in liwc_dict:
#                         liwc_dict[cat_name] = []
#                     liwc_dict[cat_name].append(word)
#                     print(f"Word: {word}, Category: {cat_name}")  # Debugging output
    
#     # Initialize a list to store LIWC results for each text
#     liwc_results = []
    
#     # Function to apply LIWC analysis on a single text
#     def analyze_text(text):
#         text_words = text.lower().split()
#         text_word_count = len(text_words)
#         category_counts = {cat: 0 for cat in liwc_dict}
        
#         # Count occurrences of each category in the text
#         for word in text_words:
#             for cat, words in liwc_dict.items():
#                 if word in words:
#                     category_counts[cat] += 1
        
#         # Calculate percentages for each category
#         category_percentages = {cat: (count / text_word_count) * 100 for cat, count in category_counts.items()}
#         return category_percentages

#     # Apply LIWC analysis to each row of text
#     for text in combined_df['text']:
#         liwc_results.append(analyze_text(text))
    
#     # Convert the results into a DataFrame
#     liwc_df = pd.DataFrame(liwc_results)
    
#     # Compute the average of each LIWC category
#     liwc_avg = liwc_df.mean()
    
#     # Convert to DataFrame to write to Parquet
#     liwc_avg_df = pd.DataFrame([liwc_avg])

#     # Write the result to a new Parquet file
#     liwc_avg_df.to_parquet('liwc.parquet', index=False)

#     return liwc_avg_df


# def save_comment_percentage_metrics(
#     posts_file,
#     comments_file,
#     precovid_posts_file,
#     precovid_comments_file,
#     city='',
#     resample_unit='W',  # Default is weekly ('W')
#     output_folder='metrics'
# ):
#     import os
#     import pandas as pd

#     def process_data(posts_file, comments_file):
#         # Load posts and comments
#         print(f"Loading posts data from {posts_file}...")
#         posts = pd.read_parquet(posts_file, columns=['id', 'created_utc'])
#         print(f"Number of posts: {len(posts)}")

#         print(f"Loading comments data from {comments_file}...")
#         comments = pd.read_parquet(comments_file, columns=['parent_id', 'created_utc'])
#         print(f"Number of comments: {len(comments)}")

#         # Convert timestamps to datetime
#         print("Converting timestamps...")
#         posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
#         comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc'], unit='s')

#         # Attach 'parent_id' in posts
#         posts['parent_id'] = 't3_' + posts['id']

#         # Sort data by time
#         posts.sort_values('created_datetime_post', inplace=True)
#         comments.sort_values('created_datetime_comment', inplace=True)

#         # Group comments by parent_id
#         comment_groups = comments.groupby('parent_id')['created_datetime_comment']

#         # Check if there's at least one comment in [post_time, post_time + 7 days)
#         print("Checking comments in [post_time, post_time + 7 days) for each post...")

#         def has_comment_in_7day_window(row):
#             pid = row['parent_id']
#             post_time = row['created_datetime_post']
#             window_end = post_time + pd.Timedelta(days=7)

#             # Get all comment times for this parent_id
#             if pid not in comment_groups.groups:
#                 return False
#             c_times = comment_groups.get_group(pid)

#             # Check for comments in the time window
#             in_window = c_times[(c_times >= post_time) & (c_times < window_end)]
#             return len(in_window) > 0

#         posts['has_comment_7day'] = posts.apply(has_comment_in_7day_window, axis=1)

#         # Resample by post creation time to compute the percentage
#         print(f"Resampling by '{resample_unit}' to compute the percentage of posts with 7-day comments...")
#         posts.set_index('created_datetime_post', inplace=True)
#         total_posts = posts.resample(resample_unit).size()
#         commented_posts = posts[posts['has_comment_7day']].resample(resample_unit).size()
#         percentage_with_comments = (commented_posts / total_posts) * 100

#         return percentage_with_comments

#     # Process pre-COVID and COVID data
#     print("Processing data...")
#     precovid_percentage = process_data(precovid_posts_file, precovid_comments_file)
#     precovid_percentage = precovid_percentage['2019-07-01':'2019-12-31']  # Oct to Dec 2019
#     main_percentage = process_data(posts_file, comments_file)

#     # Combine the datasets into one time series
#     print("Combining datasets and ensuring continuous time index...")
#     combined_percentage = pd.concat([precovid_percentage, main_percentage])

#     # Reindex to ensure a continuous time index
#     full_index = pd.date_range(
#         start=combined_percentage.index.min(),
#         end=combined_percentage.index.max(),
#         freq=resample_unit
#     )
#     combined_percentage = combined_percentage.reindex(full_index)

#     # Interpolate to fill missing values
#     combined_percentage = combined_percentage.interpolate()

#     # Prepare the metrics table
#     print("Preparing metrics table...")
#     metrics_table = pd.DataFrame(
#         {'Percentage of Posts with Comments': combined_percentage.values},
#         index=combined_percentage.index.strftime('%d/%m/%Y')
#     ).transpose()

#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#     output_path = os.path.join(output_folder, f"{city.lower()}_metrics.parquet")

#     # Save metrics table
#     print(f"Saving metrics to {output_path}...")
#     metrics_table.to_parquet(output_path, index=True)

#     print("Metrics saved successfully.")

def get_raw_post_count_metrics(
    posts_file,
    comments_file,         # Not used in this version
    precovid_posts_file,
    precovid_comments_file,  # Not used in this version
    city='',
    resample_unit='W'  # Default is weekly ('W')
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
        # Load posts data
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc'])

        # Convert timestamps to datetime
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')

        # Sort data by time
        posts.sort_values('created_datetime', inplace=True)

        # Set the datetime as index for resampling
        posts.set_index('created_datetime', inplace=True)

        # Resample by post creation time to count the number of posts
        post_counts = posts.resample(resample_unit).size()
        return post_counts

    # Process pre-COVID data (restricting date range if needed)
    precovid_counts = process_posts(precovid_posts_file)
    # For example, here we restrict the pre-COVID period to 2016-01-01 through 2019-12-31
    precovid_counts = precovid_counts['2016-01-01':'2019-12-31']

    # Process main (COVID) data
    main_counts = process_posts(posts_file)

    # Combine the two time series
    combined_counts = pd.concat([precovid_counts, main_counts])

    # Reindex to ensure a continuous time index across the whole period
    full_index = pd.date_range(
        start=combined_counts.index.min(),
        end=combined_counts.index.max(),
        freq=resample_unit
    )
    combined_counts = combined_counts.reindex(full_index).fillna(0)

    # Prepare the final metrics table with dates formatted as DD/MM/YYYY
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
    resample_unit='W'  # Default is weekly ('W')
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
        # Load comments data
        comments = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        
        # Convert timestamps to datetime
        comments['created_datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
        
        # Sort the data by creation time
        comments.sort_values('created_datetime', inplace=True)
        
        # Set the datetime as the index for resampling
        comments.set_index('created_datetime', inplace=True)
        
        # Resample by the creation time to count the number of comments per period
        comment_counts = comments.resample(resample_unit).size()
        return comment_counts

    # Process the pre-COVID comments and restrict the period if necessary.
    precovid_counts = process_comments(precovid_comments_file)
    # Example: restrict pre-COVID period to 2016-01-01 through 2019-12-31.
    precovid_counts = precovid_counts['2016-01-01':'2019-12-31']

    # Process the main (COVID) comments
    main_counts = process_comments(comments_file)

    # Combine the two time series
    combined_counts = pd.concat([precovid_counts, main_counts])

    # Reindex to ensure a continuous time index across the entire period.
    full_index = pd.date_range(
        start=combined_counts.index.min(),
        end=combined_counts.index.max(),
        freq=resample_unit
    )
    combined_counts = combined_counts.reindex(full_index).fillna(0)

    # Prepare the final metrics table with dates formatted as DD/MM/YYYY.
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
    resample_unit='W'  # Default is weekly ('W')
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
        # Load posts and comments
        # print(f"Loading posts data from {posts_path}...")
        posts = pd.read_parquet(posts_path, columns=['id', 'created_utc'])
        # print(f"Number of posts: {len(posts)}")

        # print(f"Loading comments data from {comments_path}...")
        comments = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        # print(f"Number of comments: {len(comments)}")

        # Convert timestamps to datetime
        # print("Converting timestamps...")
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc'], unit='s')

        # Attach 'parent_id' in posts (the 't3_' prefix identifies a Reddit post)
        posts['parent_id'] = 't3_' + posts['id']

        # Sort data by time
        posts.sort_values('created_datetime_post', inplace=True)
        comments.sort_values('created_datetime_comment', inplace=True)

        # Group comments by parent_id
        comment_groups = comments.groupby('parent_id')['created_datetime_comment']

        # Check if there's at least one comment in [post_time, post_time + 7 days)
        # print("Checking comments in [post_time, post_time + 7 days) for each post...")

        def has_comment_in_7day_window(row):
            pid = row['parent_id']
            post_time = row['created_datetime_post']
            window_end = post_time + pd.Timedelta(days=7)

            if pid not in comment_groups.groups:
                return False
            c_times = comment_groups.get_group(pid)

            # Check for comments in the time window
            in_window = c_times[(c_times >= post_time) & (c_times < window_end)]
            return len(in_window) > 0

        posts['has_comment_7day'] = posts.apply(has_comment_in_7day_window, axis=1)

        # Resample by post creation time to compute the percentage
        # print(f"Resampling by '{resample_unit}' to compute the % of posts with 7-day comments...")
        posts.set_index('created_datetime_post', inplace=True)

        total_posts = posts.resample(resample_unit).size()
        commented_posts = posts[posts['has_comment_7day']].resample(resample_unit).size()
        percentage_with_comments = (commented_posts / total_posts) * 100

        return percentage_with_comments

    # print("Processing data...")

    # Process pre-COVID data (restricting date range to 2019-07-01 through 2019-12-31)
    precovid_percentage = process_data(precovid_posts_file, precovid_comments_file)
    precovid_percentage = precovid_percentage['2016-01-01':'2019-12-31']

    # Process main (COVID) data
    main_percentage = process_data(posts_file, comments_file)

    # Combine into one time series
    # print("Combining datasets and ensuring continuous time index...")
    combined_percentage = pd.concat([precovid_percentage, main_percentage])

    # Reindex to ensure a continuous time index
    full_index = pd.date_range(
        start=combined_percentage.index.min(),
        end=combined_percentage.index.max(),
        freq=resample_unit
    )
    combined_percentage = combined_percentage.reindex(full_index)

    # Interpolate to fill missing values
    combined_percentage = combined_percentage.interpolate()

    # Prepare the metrics table
    # print("Preparing metrics table...")
    metrics_table = pd.DataFrame(
        {'Percentage of Posts with Comments': combined_percentage.values},
        index=combined_percentage.index.strftime('%d/%m/%Y')
    ).transpose()

    # print("Data preparation complete. Returning metrics table.")
    return metrics_table


# def save_post_lifespan_timeseries(
#     posts_file,
#     comments_file,
#     precovid_posts_file,
#     precovid_comments_file,
#     city='',
#     aggregation='monthly',
#     units='hours',
#     statistic='mean',
#     output_folder='metrics'
# ):
#     """
#     Calculates the post-lifespan time series for both pre-COVID and main datasets,
#     combines them, and appends (or creates) the result to a Parquet file.

#     Parameters:
#     -----------
#     posts_file : str
#         Path to the main posts Parquet file containing 'id' and 'created_utc'.
#     comments_file : str
#         Path to the main comments Parquet file containing 'parent_id' and 'created_utc'.
#     precovid_posts_file : str
#         Path to the pre-COVID posts Parquet file (e.g., 2019-07-01 to 2019-12-31).
#     precovid_comments_file : str
#         Path to the pre-COVID comments Parquet file.
#     city : str
#         String to include in the output file name.
#     aggregation : str
#         Time interval for aggregation ('daily', 'weekly', 'monthly', 'yearly').
#     units : str
#         Time units for lifespan: 'seconds', 'minutes', 'hours', or 'days'.
#     statistic : str
#         Statistic to compute: 'mean', 'median', or 'count'.
#     output_folder : str
#         Folder where the output Parquet file will be saved.

#     Returns:
#     --------
#     None
#         (Saves Parquet file to disk, either creating or appending to it.)
#     """

#     import os
#     import numpy as np
#     import pandas as pd

#     # ----------------------------------
#     # Helper function to process a pair of posts/comments
#     # and return an aggregated timeseries for post lifespans
#     # ----------------------------------
#     def process_data(posts_path, comments_path, aggregation, units, statistic):
#         """Process posts and comments to compute the aggregated post lifespan time series."""

#         # Load posts
#         posts_df = pd.read_parquet(posts_path, columns=['id', 'created_utc'])
#         # Convert posts created_utc to datetime
#         posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')

#         # Load comments
#         comments_df = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
#         # Clean parent_id (remove t1_, t2_, t3_, etc.)
#         comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)
#         # Convert comments created_utc to datetime
#         comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')

#         # Merge: match comments to their post by ID
#         merged_df = comments_df.merge(
#             posts_df,
#             left_on='parent_id_clean',
#             right_on='id',
#             how='inner',
#             suffixes=('_comment', '_post')
#         )

#         # Group by post ID to get min/max comment time
#         lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
#         lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

#         # Compute lifespan in seconds (max - min)
#         lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()

#         # Exclude negative lifespans if any appear
#         lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

#         # Convert lifespan to requested units
#         if units == 'seconds':
#             lifespan_df['lifespan'] = lifespan_df['lifespan_seconds']
#         elif units == 'minutes':
#             lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 60
#         elif units == 'hours':
#             lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 3600
#         elif units == 'days':
#             lifespan_df['lifespan'] = lifespan_df['lifespan_seconds'] / 86400
#         else:
#             raise ValueError(f"Invalid units '{units}'. Choose from ['seconds', 'minutes', 'hours', 'days'].")

#         # Exclude lifespans of 0 (posts with only one comment time)
#         lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]

#         # Merge back to get the post creation time
#         lifespan_df = lifespan_df.merge(
#             posts_df[['id', 'created_utc']],
#             left_on='post_id',
#             right_on='id',
#             how='left'
#         )
#         lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
#         lifespan_df.set_index('post_created_time', inplace=True)
#         lifespan_df.drop('id', axis=1, inplace=True)

#         # Determine the resampling rule based on aggregation
#         valid_aggregations = ['daily', 'weekly', 'monthly', 'yearly']
#         if aggregation not in valid_aggregations:
#             raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from {valid_aggregations}.")

#         if aggregation == 'daily':
#             resample_rule = 'D'
#         elif aggregation == 'weekly':
#             resample_rule = 'W'
#         elif aggregation == 'monthly':
#             resample_rule = 'M'
#         elif aggregation == 'yearly':
#             resample_rule = 'Y'

#         valid_statistics = ['mean', 'median', 'count']
#         if statistic not in valid_statistics:
#             raise ValueError(f"Invalid statistic '{statistic}'. Choose from {valid_statistics}.")

#         # Apply aggregation
#         if statistic in ['mean', 'median']:
#             aggregated_series = lifespan_df['lifespan'].resample(resample_rule).agg(statistic)
#         else:  # statistic == 'count'
#             aggregated_series = lifespan_df['lifespan'].resample(resample_rule).count()

#         return aggregated_series

#     # ----------------------------------
#     # Main logic to process pre-COVID and main data
#     # ----------------------------------
#     print("Processing post lifespan time series...")

#     # 1) Pre-COVID
#     precovid_lifespan = process_data(
#         precovid_posts_file,
#         precovid_comments_file,
#         aggregation=aggregation,
#         units=units,
#         statistic=statistic
#     )
#     # Restrict to 2019-07-01 to 2019-12-31
#     precovid_lifespan = precovid_lifespan['2019-07-01':'2019-12-31']

#     # 2) Main dataset
#     main_lifespan = process_data(
#         posts_file,
#         comments_file,
#         aggregation=aggregation,
#         units=units,
#         statistic=statistic
#     )

#     # Combine into one Series
#     combined_lifespan = pd.concat([precovid_lifespan, main_lifespan])

#     # Map aggregation to a pandas frequency alias
#     agg_to_freq = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'yearly': 'Y'}
#     freq_alias = agg_to_freq.get(aggregation, 'M')

#     # Reindex to get a continuous timeline
#     full_index = pd.date_range(
#         start=combined_lifespan.index.min(),
#         end=combined_lifespan.index.max(),
#         freq=freq_alias
#     )
#     combined_lifespan = combined_lifespan.reindex(full_index)

#     # Interpolate missing values if you want a smooth curve
#     combined_lifespan = combined_lifespan.interpolate()

#     # Prepare final DataFrame
#     if statistic in ['mean', 'median']:
#         col_name = f'Post Lifespan ({statistic.capitalize()} in {units})'
#     else:  # statistic == 'count'
#         col_name = 'Number of Posts'

#     # Make a single-row table (consistent with "transposed" style)
#     metrics_table = pd.DataFrame(
#         {col_name: combined_lifespan.values},
#         index=combined_lifespan.index.strftime('%d/%m/%Y')
#     ).transpose()

#     # ----------------------------------
#     # Append to an existing Parquet if it exists
#     # ----------------------------------
#     os.makedirs(output_folder, exist_ok=True)
#     output_path = os.path.join(output_folder, f"{city.lower()}_post_lifespan_metrics.parquet")

#     if os.path.exists(output_path):
#         print(f"Appending to existing Parquet file: {output_path}")
#         existing_df = pd.read_parquet(output_path)

#         # Combine the old and new data so that:
#         # - The row(s) from `metrics_table` are added/updated
#         # - The columns (dates) from `metrics_table` are added/updated
#         #
#         # If the same row index exists, we update those cells; otherwise, we add a new row.
#         # If the same column index exists, we overwrite or merge.

#         # Make a copy so we don't accidentally mutate existing_df
#         combined_df = existing_df.copy()

#         for metric_row in metrics_table.index:
#             # If this metric row already exists, just update its columns
#             if metric_row in combined_df.index:
#                 # Reindex to ensure new columns exist in combined_df
#                 all_columns = combined_df.columns.union(metrics_table.columns)
#                 combined_df = combined_df.reindex(columns=all_columns)
#                 # Overwrite the values for these columns
#                 combined_df.loc[metric_row, metrics_table.columns] = metrics_table.loc[metric_row, metrics_table.columns]
#             else:
#                 # If this row doesn't exist, just append it
#                 combined_df = pd.concat([combined_df, metrics_table.loc[[metric_row]]], axis=0)

#         # Sort columns by date if desired (since they are string dates, you might want a custom sort)
#         # For a simple lexicographical sort of DD/MM/YYYY:
#         # combined_df = combined_df.reindex(sorted(combined_df.columns, key=lambda d: pd.to_datetime(d, format='%d/%m/%Y')), axis=1)

#         # Finally, overwrite the existing Parquet
#         combined_df.to_parquet(output_path)
#     else:
#         print(f"Creating new Parquet file: {output_path}")
#         metrics_table.to_parquet(output_path)

#     print("Post lifespan metrics saved/appended successfully.")


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
    
    # ----------------------------------
    # Helper function to process a pair of posts/comments
    # and return an aggregated timeseries for post lifespans
    # ----------------------------------
    def process_data(posts_path, comments_path, aggregation, units, statistic):
        """Process posts and comments to compute the aggregated post-lifespan time series."""

        # Load posts
        posts_df = pd.read_parquet(posts_path, columns=['id', 'created_utc'])
        # Convert posts created_utc to datetime
        posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit='s')

        # Load comments
        comments_df = pd.read_parquet(comments_path, columns=['parent_id', 'created_utc'])
        # Clean parent_id (remove t1_, t2_, t3_, etc.)
        comments_df['parent_id_clean'] = comments_df['parent_id'].str.replace(r'^t[1-3]_', '', regex=True)
        # Convert comments created_utc to datetime
        comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')

        # Merge: match comments to their post by ID
        merged_df = comments_df.merge(
            posts_df,
            left_on='parent_id_clean',
            right_on='id',
            how='inner',
            suffixes=('_comment', '_post')
        )

        # Group by post ID to get min/max comment time
        lifespan_df = merged_df.groupby('id')['created_utc_comment'].agg(['min', 'max']).reset_index()
        lifespan_df.rename(columns={'id': 'post_id'}, inplace=True)

        # Compute lifespan in seconds (max - min)
        lifespan_df['lifespan_seconds'] = (lifespan_df['max'] - lifespan_df['min']).dt.total_seconds()

        # Exclude negative lifespans if any appear
        lifespan_df = lifespan_df[lifespan_df['lifespan_seconds'] >= 0]

        # Convert lifespan to requested units
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

        # Exclude lifespans of 0 (posts with only one comment time)
        lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]

        # Merge back to get the post creation time
        lifespan_df = lifespan_df.merge(
            posts_df[['id', 'created_utc']],
            left_on='post_id',
            right_on='id',
            how='left'
        )
        lifespan_df.rename(columns={'created_utc': 'post_created_time'}, inplace=True)
        lifespan_df.set_index('post_created_time', inplace=True)
        lifespan_df.drop('id', axis=1, inplace=True)

        # Determine the resampling rule based on aggregation
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

        # Apply aggregation
        if statistic in ['mean', 'median']:
            aggregated_series = lifespan_df['lifespan'].resample(resample_rule).agg(statistic)
        else:  # statistic == 'count'
            aggregated_series = lifespan_df['lifespan'].resample(resample_rule).count()

        return aggregated_series

    # ----------------------------------
    # Main logic to process pre-COVID and main data
    # ----------------------------------
    # print("Processing post lifespan time series...")

    # Process pre-COVID data
    precovid_lifespan = process_data(
        precovid_posts_file,
        precovid_comments_file,
        aggregation=aggregation,
        units=units,
        statistic=statistic
    )
    # Restrict to (for example) 2019-07-01 to 2019-12-31
    precovid_lifespan = precovid_lifespan['2016-01-01':'2019-12-31']

    # Process main dataset
    main_lifespan = process_data(
        posts_file,
        comments_file,
        aggregation=aggregation,
        units=units,
        statistic=statistic
    )

    # Combine into one Series
    combined_lifespan = pd.concat([precovid_lifespan, main_lifespan])

    # Map aggregation to a pandas frequency alias
    agg_to_freq = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'yearly': 'Y'}
    freq_alias = agg_to_freq.get(aggregation, 'M')

    # Reindex to get a continuous timeline
    full_index = pd.date_range(
        start=combined_lifespan.index.min(),
        end=combined_lifespan.index.max(),
        freq=freq_alias
    )
    combined_lifespan = combined_lifespan.reindex(full_index)

    # Interpolate missing values for a smoother curve
    combined_lifespan = combined_lifespan.interpolate()

    # Prepare the final DataFrame
    if statistic in ['mean', 'median']:
        col_name = f'Post Lifespan ({statistic.capitalize()} in {units})'
    else:  # statistic == 'count'
        col_name = 'Number of Posts'

    # Create a single-row table, with date columns and one row for the metric
    metrics_table = pd.DataFrame(
        {col_name: combined_lifespan.values},
        index=combined_lifespan.index.strftime('%d/%m/%Y')
    ).transpose()

    # print("Post lifespan metrics computed. Returning DataFrame...")
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

        # Rename raw timestamps to *_post / *_comment
        posts.rename(columns={'created_utc': 'created_utc_post'}, inplace=True)
        comments.rename(columns={'created_utc': 'created_utc_comment'}, inplace=True)

        # Convert timestamps to proper datetime columns
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc_comment'], unit='s')

        # Add the Reddit 't3_' prefix to post IDs
        posts['parent_id'] = 't3_' + posts['id']

        # Merge comments to posts
        merged = comments.merge(
            posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
            on='parent_id',
            how='inner'
        )

        # Sort by parent_id and comment time so we can pick the first comment per post
        merged_sorted = merged.sort_values(['parent_id', 'created_datetime_comment'])
        first_comments = merged_sorted.groupby('parent_id').first().reset_index()

        # Calculate response time using datetime columns
        # => results in Timedelta, convert to float seconds via .dt.total_seconds()
        first_comments['response_time_seconds'] = (
            first_comments['created_datetime_comment'] - first_comments['created_datetime_post']
        ).dt.total_seconds()

        # Convert to desired time_diff_unit
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

        # Resample on the post creation time
        first_comments.set_index('created_datetime_post', inplace=True)
        avg_response = first_comments['response_time'].resample(time_unit).mean()

        return avg_response

    print("Processing pre-COVID data...")
    precovid_series = process_data(precovid_posts_file, precovid_comments_file)
    # Optional: restrict the pre-COVID data to a particular date range if desired
    precovid_series = precovid_series['2016-01-01':'2019-12-31']

    print("Processing main data...")
    main_series = process_data(posts_file, comments_file)

    # Combine pre-COVID and main data
    combined_series = pd.concat([precovid_series, main_series])

    # Ensure a continuous date range
    full_index = pd.date_range(
        start=combined_series.index.min(),
        end=combined_series.index.max(),
        freq=time_unit  # e.g., 'W' for weekly
    )
    combined_series = combined_series.reindex(full_index)

    # Interpolate any missing points
    combined_series = combined_series.interpolate()

    # Create a single-row DataFrame
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
        
        # Rename timestamps for clarity
        posts.rename(columns={'created_utc': 'created_utc_post'}, inplace=True)
        comments.rename(columns={'created_utc': 'created_utc_comment'}, inplace=True)
        
        # Convert timestamps to datetime
        posts['created_datetime_post'] = pd.to_datetime(posts['created_utc_post'], unit='s')
        comments['created_datetime_comment'] = pd.to_datetime(comments['created_utc_comment'], unit='s')
        
        # Add the Reddit 't3_' prefix to match the comment parent IDs
        posts['parent_id'] = 't3_' + posts['id']
        
        # Merge comments with posts based on parent_id
        merged = comments.merge(
            posts[['parent_id', 'created_datetime_post', 'created_utc_post']],
            on='parent_id',
            how='inner'
        )
        
        # Sort by post and comment time to identify the first comment per post
        merged_sorted = merged.sort_values(['parent_id', 'created_datetime_comment'])
        first_comments = merged_sorted.groupby('parent_id').first().reset_index()
        
        # Calculate response time (as a timedelta in seconds)
        first_comments['response_time_seconds'] = (
            first_comments['created_datetime_comment'] - first_comments['created_datetime_post']
        ).dt.total_seconds()
        
        # Apply cutoff: Exclude responses that took more than 24 hours (86400 seconds)
        first_comments = first_comments[first_comments['response_time_seconds'] <= 86400]
        
        # Convert response time to the desired unit
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
        
        # Resample on the post creation time to compute the average response time per period
        first_comments.set_index('created_datetime_post', inplace=True)
        avg_response = first_comments['response_time'].resample(time_unit).mean()
        
        return avg_response

    print("Processing pre-COVID data...")
    precovid_series = process_data(precovid_posts_file, precovid_comments_file)
    # Optionally restrict pre-COVID data to a desired time range
    precovid_series = precovid_series['2016-01-01':'2019-12-31']
    
    print("Processing main data...")
    main_series = process_data(posts_file, comments_file)
    
    # Combine the pre-COVID and main datasets
    combined_series = pd.concat([precovid_series, main_series])
    
    # Ensure a continuous date range
    full_index = pd.date_range(
        start=combined_series.index.min(),
        end=combined_series.index.max(),
        freq=time_unit
    )
    combined_series = combined_series.reindex(full_index)
    
    # Interpolate any missing values
    combined_series = combined_series.interpolate()
    
    # Set the row label based on the time_diff_unit
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
    
    # Create a single-row DataFrame with dates as column headers (formatted as DD/MM/YYYY)
    metrics_df = pd.DataFrame(
        {row_label: combined_series.values},
        index=combined_series.index.strftime('%d/%m/%Y')
    ).transpose()
    
    print("Response time data (with 24-hour cutoff) for pre-COVID and main datasets combined. Returning DataFrame...")
    return metrics_df


def save_timeseries_metrics_for_cities(city_dict):
    # Create the output directory if it doesn't exist
    output_dir = "../metrics"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        # Generate standardized file paths using the city key.
        # Removing spaces and converting to lowercase ensures consistency.
        city_lower = city_key.lower().replace(" ", "")
        submissions_path = f"../covid_data_parquet/{city_lower}_submissions.parquet"
        comments_path = f"../covid_data_parquet/{city_lower}_comments.parquet"
        precovid_submissions_path = f"../precovid_data_parquet/{city_lower}_submissions.parquet"
        precovid_comments_path = f"../precovid_data_parquet/{city_lower}_comments.parquet"
        prophet_train_submissions_path = f"../prophet_train_parquet/{city_lower}_submissions.parquet"
        prophet_train_comments_path = f"../prophet_train_parquet/{city_lower}_comments.parquet"

        # save_comment_percentage_metrics(
        #     submissions_path,
        #     comments_path,
        #     precovid_submissions_path,
        #     precovid_comments_path,
        #     city=city_lower,
        #     resample_unit='W',  # 'W' for weekly, 'D' for daily, 'M' for monthly, etc.
        #     output_folder=output_dir
        # )

        post_count_metrics = get_raw_post_count_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'  # Default is weekly ('W')
        )

        comment_count_metrics = get_raw_comment_count_metrics(
            submissions_path,
            comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'  # Default is weekly ('W')
        )

        comment_percentage_metrics = get_comment_percentage_metrics(
            submissions_path,
            comments_path,
            # precovid_submissions_path,
            # precovid_comments_path,
            prophet_train_submissions_path,
            prophet_train_comments_path,
            city=city_lower,
            resample_unit='W'  # Default is weekly ('W')
        )

        lifespan_metrics = get_post_lifespan_timeseries(
            submissions_path,
            comments_path,
            # precovid_submissions_path,
            # precovid_comments_path,
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
            # precovid_submissions_path,
            # precovid_comments_path,
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

        # 3) Concatenate the two single-row DataFrames into one
        #    Each metric is a separate row; columns are dates.
        combined_metrics = pd.concat([post_count_metrics, comment_count_metrics, comment_percentage_metrics, lifespan_metrics, response_metrics, cutoff_response_metrics], axis=0)
        output_path = os.path.join(output_dir, f"{city_lower}_metrics.parquet")
        combined_metrics.to_parquet(output_path)


def cluster_cities_with_dtw(
    city_files_folder,
    output_file,
    resample_unit='W',
    max_clusters=10,
    method='kmedoids'  # Options: 'kmedoids' or 'hierarchical'
):
    """
    Performs DTW clustering on city time series data stored in parquet files.

    Parameters:
    - city_files_folder (str): Path to the folder containing city parquet files.
    - output_file (str): Path to save the combined results as a parquet file.
    - resample_unit (str): Resampling frequency for time series (default is weekly).
    - max_clusters (int): Maximum number of clusters to test for finding the optimal number of clusters.
    - method (str): Clustering method ('kmedoids' or 'hierarchical').

    Saves:
    - A combined parquet file with the clustering results for all cities.
    """

    def load_city_data(file_path):
        df = pd.read_parquet(file_path)
        df.index = pd.to_datetime(df.index)
        df = df.resample(resample_unit).mean().interpolate()
        return df

    def calculate_dtw_distance_matrix(time_series_data):
        n = len(time_series_data)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance_matrix[i, j] = dtw.distance(time_series_data[i], time_series_data[j])
                distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric
        return distance_matrix

    def find_optimal_clusters(distance_matrix):
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmedoids = KMedoids(n_clusters=k, metric="precomputed").fit(distance_matrix)
            labels = kmedoids.labels_
            silhouette_scores.append(silhouette_score(distance_matrix, labels, metric="precomputed"))

        optimal_k = np.argmax(silhouette_scores) + 2  # +2 because k starts at 2
        return optimal_k, silhouette_scores

    # Gather city parquet files
    city_files = [os.path.join(city_files_folder, f) for f in os.listdir(city_files_folder) if f.endswith('.parquet')]
    city_names = [os.path.splitext(os.path.basename(f))[0] for f in city_files]

    # Load all city time series
    print("Loading city data...")
    city_data = [load_city_data(f).values.flatten() for f in city_files]

    # Calculate DTW distance matrix
    print("Calculating DTW distance matrix...")
    distance_matrix = calculate_dtw_distance_matrix(city_data)

    # Determine the optimal number of clusters
    print("Finding the optimal number of clusters...")
    optimal_clusters, silhouette_scores = find_optimal_clusters(distance_matrix)

    # Perform clustering
    if method == 'kmedoids':
        print(f"Clustering cities using K-Medoids with {optimal_clusters} clusters...")
        clustering_model = KMedoids(n_clusters=optimal_clusters, metric="precomputed")
        cluster_labels = clustering_model.fit_predict(distance_matrix)
    elif method == 'hierarchical':
        print(f"Clustering cities using hierarchical clustering with {optimal_clusters} clusters...")
        linkage_matrix = linkage(distance_matrix, method="ward")
        cluster_labels = fcluster(linkage_matrix, optimal_clusters, criterion="maxclust")
    else:
        raise ValueError("Invalid clustering method. Choose 'kmedoids' or 'hierarchical'.")

    # Save results
    print("Saving clustering results...")
    results_df = pd.DataFrame({
        "City": city_names,
        "Cluster": cluster_labels
    })

    # Save to parquet file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_parquet(output_file, index=False)

    print(f"Clustering results saved to {output_file}")


# def cluster_cities_with_dtw_pyclustering(
#     city_files_folder,
#     output_file,
#     resample_unit='W',
#     max_clusters=10,
#     n_jobs=-1
# ):
#     """
#     Performs DTW clustering on city time series data stored in parquet files using `pyclustering`.

#     Parameters:
#     - city_files_folder (str): Path to the folder containing city parquet files.
#     - output_file (str): Path to save the combined results as a parquet file.
#     - resample_unit (str): Resampling frequency for time series (default is weekly).
#     - max_clusters (int): Maximum number of clusters to test for finding the optimal number of clusters.

#     Saves:
#     - A combined parquet file with the clustering results for all cities.
#     """

#     def load_city_data(file_path):
#         """
#         Loads a city's metrics parquet file, extracts datetime values from column headings, 
#         and converts it to a resampled and interpolated time series.
#         """
#         # Load data
#         df = pd.read_parquet(file_path)
        
#         # Ensure column headers are datetime values
#         try:
#             df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
#         except Exception as e:
#             raise ValueError(f"Error converting column headers to datetime in {file_path}: {e}")
        
#         # Transpose the DataFrame to make datetime the index
#         df = df.T
#         df.index.name = "datetime"

#         df = df.loc["2020-01-01":"2021-12-31"]

#         # Resample and interpolate
#         df = df.resample(resample_unit).mean().interpolate()
#         return df


#     def calculate_dtw_distance_matrix(time_series_data):
#         n = len(time_series_data)
#         distance_matrix = np.zeros((n, n))
#         for i in range(n):
#             for j in range(i + 1, n):
#                 distance_matrix[i, j] = dtw.distance(time_series_data[i], time_series_data[j])
#                 distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric
#         return distance_matrix
    
#     def compute_dtw_row(i, time_series_list):
#         """
#         Computes DTW distances from time series i to all subsequent series j>i
#         to avoid double computation (since the distance matrix is symmetric).
#         """
#         n = len(time_series_list)
#         row = np.zeros(n)
#         for j in range(i + 1, n):
#             row[j] = dtw.distance(time_series_list[i], time_series_list[j])
#         return i, row

#     def calculate_dtw_distance_matrix_parallel(time_series_list, n_jobs=-1):
#         """
#         Computes the pairwise DTW distance matrix in parallel using joblib.
#         Returns an NxN numpy array of DTW distances.
#         """
#         n = len(time_series_list)
#         distance_matrix = np.zeros((n, n))
        
#         # Parallel computation: each 'compute_dtw_row' handles one row (i).
#         results = Parallel(n_jobs=n_jobs)(
#             delayed(compute_dtw_row)(i, time_series_list) for i in range(n)
#         )
        
#         # Fill the upper and lower triangular parts of the matrix.
#         for i, row in results:
#             distance_matrix[i, i+1:] = row[i+1:]
#             distance_matrix[i+1:, i] = row[i+1:]
        
#         return distance_matrix

#     def find_optimal_clusters(distance_matrix):
#         # Try clustering with different numbers of clusters and compute Silhouette scores
#         silhouette_scores = []
#         for k in range(2, max_clusters + 1):
#             initial_medoids = list(range(k))  # Use the first `k` points as initial medoids
#             kmedoids_instance = kmedoids(distance_matrix.tolist(), initial_medoids)
#             kmedoids_instance.process()
#             clusters = kmedoids_instance.get_clusters()

#             # Compute silhouette score
#             labels = np.zeros(len(distance_matrix), dtype=int)
#             for cluster_id, cluster in enumerate(clusters):
#                 for city_index in cluster:
#                     labels[city_index] = cluster_id

#             from sklearn.metrics import silhouette_score
#             silhouette_scores.append(silhouette_score(distance_matrix, labels, metric="precomputed"))

#         optimal_k = np.argmax(silhouette_scores) + 2  # +2 because k starts at 2
#         return optimal_k, silhouette_scores
    
#     output_dir = os.path.dirname(output_file) or "."  # Use current directory if none provided
#     os.makedirs(output_dir, exist_ok=True)

#     # Gather city parquet files
#     city_files = [os.path.join(city_files_folder, f) for f in os.listdir(city_files_folder) if f.endswith('.parquet')]
#     city_names = [os.path.splitext(os.path.basename(f))[0] for f in city_files]

#     # Load all city time series
#     print("Loading city data...")
#     city_data = [load_city_data(f).values.flatten() for f in city_files]

#     # Calculate DTW distance matrix
#     print("Calculating DTW distance matrix...")
#     # distance_matrix = calculate_dtw_distance_matrix(city_data)
#     distance_matrix = calculate_dtw_distance_matrix_parallel(city_data, n_jobs)

#     # Determine the optimal number of clusters
#     print("Finding the optimal number of clusters...")
#     optimal_clusters, silhouette_scores = find_optimal_clusters(distance_matrix)

#     # Perform clustering using optimal number of clusters
#     print(f"Clustering cities using K-Medoids with {optimal_clusters} clusters...")
#     initial_medoids = list(range(optimal_clusters))  # Initialize with the first few indices
#     kmedoids_instance = kmedoids(distance_matrix.tolist(), initial_medoids)
#     kmedoids_instance.process()
#     clusters = kmedoids_instance.get_clusters()

#     # Assign cluster labels
#     cluster_labels = np.zeros(len(city_names), dtype=int)
#     for cluster_id, cluster_cities in enumerate(clusters):
#         for city_index in cluster_cities:
#             cluster_labels[city_index] = cluster_id

#     # Save results
#     print("Saving clustering results...")
#     results_df = pd.DataFrame({
#         "City": [name.replace("_metrics", "") for name in city_names],
#         "Cluster": cluster_labels
#     })


#     # Save to parquet file
#     # os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     results_df.to_parquet(output_file, index=False)

#     print(f"Clustering results saved to {output_file}")


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

    # ----------------------- Data Loading ----------------------- #
    def load_city_data(file_path, metric):
        """
        Loads a city's metrics parquet file, extracts datetime values from column headings, 
        and converts it to a resampled and interpolated time series for the chosen metric only.
        """
        # Load data
        df = pd.read_parquet(file_path)
        
        # Ensure column headers are datetime
        try:
            df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
        except Exception as e:
            raise ValueError(f"Error converting column headers to datetime in {file_path}: {e}")
        
        # Transpose the DataFrame to make datetime the index
        # Now, df's rows = dates, df's columns = metric names
        df = df.T
        df.index.name = "datetime"

        df.index = pd.to_datetime(df.index)  # Ensure DatetimeIndex
        df = df.sort_index()  # Ensure it's sorted

        # Print to debug
        print(df.index.min(), df.index.max())

        # Filter date range
        df = df.loc["2020-01-01":"2021-12-31"]

        # --- Select only the requested metric ---
        # NOTE: The 'metric' name must match exactly the column name in your data
        # e.g. "Percentage of Posts with Comments"
        # If your data has a slightly different naming scheme, adjust accordingly.
        if metric not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in file columns. Available metrics: {list(df.columns)}"
            )
        
        df = df[[metric]]  # Keep only the chosen metric as a single column DataFrame

        # Resample and interpolate
        df = df.resample(resample_unit).mean().interpolate()

        # Flatten to a 1D array
        return df.values.flatten()
    # ------------------------------------------------------------ #

    # --------------- Parallelized DTW Computation --------------- #
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
        
        # Parallel computation: each 'compute_dtw_row' handles one row (i).
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw_row)(i, time_series_list) for i in range(n)
        )
        
        # Fill the upper and lower triangular parts of the matrix.
        for i, row in results:
            distance_matrix[i, i+1:] = row[i+1:]
            distance_matrix[i+1:, i] = row[i+1:]
        
        return distance_matrix
    # ------------------------------------------------------------ #

    # ----------------- Finding Optimal Clusters ----------------- #
    def find_optimal_clusters(distance_matrix):
        """
        Try clustering with k=2..max_clusters, compute silhouette scores,
        pick k with the highest silhouette.
        """
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            # Simple init: first k points as medoids
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

        # k starts at 2, so we add 2 to the index of the max
        optimal_k = np.argmax(silhouette_scores) + 2
        return optimal_k, silhouette_scores
    # ------------------------------------------------------------ #

    # ---------------------- Main Pipeline ------------------------ #
    print("Gathering city files...")
    # city_files = [
    #     os.path.join(city_files_folder, f)
    #     for f in os.listdir(city_files_folder)
    #     if f.endswith('.parquet')
    # ]
    # city_names = [os.path.splitext(os.path.basename(f))[0] for f in city_files]

    # if selected_cities is not None:
    #     selected_city_files = []
    #     selected_city_names = []
    #     for f, name in zip(city_files, city_names):
    #         base_name = name.replace("_metrics", "")
    #         if base_name in selected_cities:
    #             selected_city_files.append(f)
    #             selected_city_names.append(name)
    #     if not selected_city_files:
    #         raise ValueError("None of the selected cities were found in the provided folder.")
    #     else:
    #         city_files = selected_city_files
    #         city_names = selected_city_names

    # print(city_files)
    # print(city_names)

    all_files = [
        os.path.join(city_files_folder, f)
        for f in os.listdir(city_files_folder)
        if f.endswith('.parquet')
    ]
    # Derive city names from file names (without extension)
    all_city_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

    # If selected cities are provided, convert them to lowercase for robust matching.
    if selected_cities is not None:
        if isinstance(selected_cities, str):
            selected_cities = [selected_cities]
        selected_cities = [s.lower() for s in selected_cities]

        filtered_files = []
        filtered_names = []
        for f, name in zip(all_files, all_city_names):
            # Remove '_metrics' from the name and compare in lowercase
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

    # Load the time series for each city, for the specified metric
    print(f"Loading city data for metric='{metric_name}'...")
    city_data = [load_city_data(f, metric_name) for f in city_files]

    # Calculate DTW distance matrix (in parallel)
    print("Calculating DTW distance matrix...")
    distance_matrix = calculate_dtw_distance_matrix_parallel(city_data, n_jobs=n_jobs)

    # Find the optimal number of clusters
    print("Finding the optimal number of clusters...")
    optimal_clusters, silhouette_scores = find_optimal_clusters(distance_matrix)
    print(f"Optimal number of clusters by silhouette = {optimal_clusters}")

    # Perform final clustering with that k
    print(f"Performing K-Medoids with k={optimal_clusters}...")
    initial_medoids = list(range(optimal_clusters))
    kmedoids_instance = kmedoids(distance_matrix.tolist(), initial_medoids)
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()

    # Assign labels
    cluster_labels = np.zeros(len(city_names), dtype=int)
    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            cluster_labels[idx] = cluster_id

    # Create a results DataFrame
    # (remove "_metrics" if you prefer shorter names)
    results_df = pd.DataFrame({
        "City": [name.replace("_metrics", "") for name in city_names],
        "Cluster": cluster_labels
    })

    # Save to parquet
    print(f"Saving clustering results to {output_file}...")
    results_df.to_parquet(output_file, index=False)
    print("Done.")

    return optimal_clusters, silhouette_scores


def cluster_cities_with_hierarchical_dtw_parallel(
    city_files_folder,
    output_file,
    resample_unit='W',
    max_clusters=10,
    linkage_method='complete',
    n_jobs=-1
):
    """
    Performs Hierarchical Clustering on city time series data stored in parquet files.
    The distance matrix is computed in parallel using DTW. The number of clusters
    is chosen to maximize the Silhouette score.
    
    Only uses data from 1st Jan 2020 through 31st Dec 2021.

    Parameters:
    - city_files_folder (str): Path to the folder containing city parquet files.
    - output_file (str): Path to save the combined results as a parquet file.
    - resample_unit (str): Resampling frequency for time series (default is weekly: 'W').
    - max_clusters (int): Maximum number of clusters to test for finding the optimal number of clusters.
    - linkage_method (str): Hierarchical linkage method ('single', 'complete', 'average', etc.).
    - n_jobs (int): Number of parallel jobs to use for DTW distance calculation.
                    Default is -1 (uses all available cores).

    Saves:
    - A combined parquet file with the clustering results for all cities.
    Returns:
    - best_k (int): The optimal number of clusters as determined by the silhouette score.
    - silhouette_scores (list of float): The silhouette scores for k in [2..max_clusters].
    """

    # ------------------------- Data Loading ------------------------- #
    def load_city_data(file_path):
        """
        Loads a city's metrics parquet file, ensures columns are datetime,
        filters rows to [2020-01-01, 2021-12-31], then resamples and interpolates.
        
        The input data is assumed to have weekly columns (or daily) in
        dd/mm/YYYY format. If columns are weekly, each column is the date for that week.
        """
        df = pd.read_parquet(file_path)
        
        # Convert columns to datetime
        try:
            df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
        except Exception as e:
            raise ValueError(f"Error converting column headers to datetime in {file_path}: {e}")
        
        # Transpose to put datetime on the index axis
        df = df.T  # Now df.index are the datetimes, df columns are metrics
        df.index.name = "datetime"

        # ---------------------------------------------------------
        # Filter to the desired date range: 2020-01-01 to 2021-12-31
        # ---------------------------------------------------------
        df = df.loc["2020-01-01":"2021-12-31"]
        
        # Resample (e.g., weekly) and interpolate
        df = df.resample(resample_unit).mean().interpolate()

        # Return as 1D array (flatten all metrics within that date range)
        return df.values.flatten()
    # --------------------------------------------------------------- #

    # -------------------- Parallelized DTW ------------------------- #
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
        
        # Parallel computation: each 'compute_dtw_row' handles one row (i).
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw_row)(i, time_series_list) for i in range(n)
        )
        
        # Fill the upper and lower triangular parts of the matrix.
        for i, row in results:
            distance_matrix[i, i+1:] = row[i+1:]
            distance_matrix[i+1:, i] = row[i+1:]
        
        return distance_matrix
    # -------------------------------------------------------------- #

    # -------------------- Main Pipeline --------------------------- #
    # Create output directory if needed
    output_dir = os.path.dirname(output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Gather city parquet files
    city_files = [
        os.path.join(city_files_folder, f)
        for f in os.listdir(city_files_folder)
        if f.endswith('.parquet')
    ]
    city_names = [os.path.splitext(os.path.basename(f))[0] for f in city_files]

    # Load time series data for each city
    print("Loading city data...")
    city_data = [load_city_data(f) for f in city_files]
    
    # Calculate the DTW distance matrix in parallel
    print("Calculating DTW distance matrix (parallel)...")
    distance_matrix = calculate_dtw_distance_matrix_parallel(city_data, n_jobs=n_jobs)

    # Convert NxN matrix -> condensed form for `scipy.linkage`
    dist_condensed = squareform(distance_matrix, checks=False)

    # Hierarchical linkage
    print(f"Performing hierarchical clustering with method='{linkage_method}'...")
    Z = linkage(dist_condensed, method=linkage_method)

    # Evaluate silhouette scores for k in [2..max_clusters]
    best_score = -1
    best_k = None
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        
        # Compute silhouette using the precomputed distance matrix
        score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Best number of clusters (by silhouette) = {best_k} with score = {best_score:.4f}")

    # Get final cluster assignment using best_k
    final_labels = fcluster(Z, best_k, criterion='maxclust')

    # Build results DataFrame
    results_df = pd.DataFrame({
        "City": [name.replace("_metrics", "") for name in city_names],
        "Cluster": final_labels
    })

    # Save results
    print("Saving clustering results...")
    results_df.to_parquet(output_file, index=False)
    print(f"Clustering results saved to {output_file}")

    # Optionally return best_k and silhouette scores
    return best_k, silhouette_scores


# def dunn_index(distance_matrix, labels):
#     """
#     Computes the Dunn index given a distance matrix and cluster labels.
#     Dunn index = (min inter-cluster distance) / (max intra-cluster distance).
#     Higher is better.
#     """
#     unique_labels = np.unique(labels)
#     # Convert to int in case labels are float
#     clusters = [np.where(labels == c)[0] for c in unique_labels]

#     # Calculate intra-cluster distances (max)
#     intra_dists = []
#     for cluster_indices in clusters:
#         if len(cluster_indices) > 1:
#             # Max distance within a cluster
#             dists = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
#             intra_dists.append(dists.max())
#         else:
#             # If only one element in cluster, no intra-distance
#             intra_dists.append(0)
#     max_intra = np.max(intra_dists)

#     # Calculate inter-cluster distances (min)
#     inter_dists = []
#     for i in range(len(clusters)):
#         for j in range(i + 1, len(clusters)):
#             # Distances between all points of cluster i and cluster j
#             dists = distance_matrix[np.ix_(clusters[i], clusters[j])]
#             inter_dists.append(dists.min())
#     min_inter = np.min(inter_dists) if inter_dists else 0

#     # Avoid division by zero
#     if max_intra == 0:
#         return 0

#     return min_inter / max_intra


# def connectivity(distance_matrix, labels, neighborhood_size=10):
#     """
#     Computes a simple 'connectivity' measure. 
#     One approach: for each sample, look at the k-nearest neighbors (k=neighborhood_size).
#     If neighbor is in a different cluster, that adds to the connectivity penalty.
#     Lower connectivity is better (more 'compact' clusters).
#     """
#     n = len(labels)
#     penalties = 0

#     for i in range(n):
#         # Distances from i to all others
#         row_dists = distance_matrix[i, :]
#         # Exclude self by setting it to a large number
#         row_dists[i] = np.inf
#         # Get the indices of the k-nearest neighbors
#         nn_indices = np.argsort(row_dists)[:neighborhood_size]

#         # If neighbor is in different cluster, add penalty
#         for nb in nn_indices:
#             if labels[nb] != labels[i]:
#                 penalties += 1

#     return penalties

# # -------------------
# # Multi-dimensional DTW distance computation
# # -------------------
# def multi_dtw_distance(series_a, series_b):
#     """
#     Example placeholder for multi-dimensional DTW distance.
#     Suppose 'series_a' and 'series_b' have shape (T, D) -> T time steps, D dimensions.
#     We'll sum up 1D-DTW distances across dimensions to get an overall distance.
#     Adjust as needed for your data/library.
#     """
#     if series_a.shape[1] != series_b.shape[1]:
#         raise ValueError("Series have different dimensionalities.")

#     # Example: sum of 1D-DTW distances
#     total_distance = 0.0
#     for dim in range(series_a.shape[1]):
#         # Using 'dtw' distance from the dtw-python library or a similar approach
#         # dist = dtw(series_a[:, dim], series_b[:, dim]).distance
#         # For demonstration (and to avoid external dependencies), just use Euclidean as a placeholder:
#         # You will replace with the actual dtw(...) call.
#         dist = np.linalg.norm(series_a[:, dim] - series_b[:, dim])
#         total_distance += dist

#     return total_distance


# def calculate_dtw_distance_matrix(time_series_data):
#     """
#     Calculates a full distance matrix using multi-dimensional DTW.
#     `time_series_data` is a list of numpy arrays. Each array can be shape (T, D).
#     """
#     n = len(time_series_data)
#     distance_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i + 1, n):
#             dist = multi_dtw_distance(time_series_data[i], time_series_data[j])
#             distance_matrix[i, j] = dist
#             distance_matrix[j, i] = dist

#     return distance_matrix


# def cluster_cities_with_dtw_hierarchical(
#     city_files_folder,
#     output_file,
#     resample_unit='W',
#     max_clusters=10
# ):
#     """
#     1) Load city data (potentially multi-dimensional).
#     2) Compute pairwise dissimilarities using multi-dimensional DTW.
#     3) Use hierarchical clustering (Ward's criterion).
#     4) For k in [2..max_clusters], compute Silhouette, Dunn index, and Connectivity.
#     5) Select the best solution (by any preferred metric) and save results.
#     """
#     # -------------------
#     # STEP 1: Load the city data
#     # -------------------
#     def load_city_data(file_path):
#         """
#         Load a city's metrics parquet file.
#         Ensure each row is a time step and each column is a dimension (if multi-dimensional).
#         """
#         df = pd.read_parquet(file_path)

#         # If columns are dates, convert them
#         # (Alternatively, your data might have time on the rows already.)
#         try:
#             df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
#         except Exception:
#             # If your data is already row-wise by time, adapt or skip this step
#             pass

#         # Transpose so that time is the row axis
#         df = df.T
#         df.index.name = "datetime"

#         # Resample & interpolate
#         df = df.resample(resample_unit).mean().interpolate()

#         # Return as a NumPy array with shape (T, D) -> T time steps, D features
#         return df.values

#     output_dir = os.path.dirname(output_file) or "."
#     os.makedirs(output_dir, exist_ok=True)

#     # Gather city parquet files
#     city_files = [
#         os.path.join(city_files_folder, f)
#         for f in os.listdir(city_files_folder)
#         if f.endswith('.parquet')
#     ]
#     city_names = [os.path.splitext(os.path.basename(f))[0] for f in city_files]

#     # Load time series data for each city
#     print("Loading city data...")
#     city_data = [load_city_data(f) for f in city_files]

#     # -------------------
#     # STEP 2: Compute DTW distance matrix
#     # -------------------
#     print("Calculating DTW distance matrix...")
#     distance_matrix = calculate_dtw_distance_matrix(city_data)

#     # -------------------
#     # STEP 3: Hierarchical Clustering (Ward's method)
#     # -------------------
#     print("Performing hierarchical clustering...")
#     # 'linkage' expects a condensed distance matrix for 'ward' method.
#     # We'll use `squareform(distance_matrix)` to convert from square to condensed form.
#     condensed_distances = squareform(distance_matrix, checks=False)
#     Z = linkage(condensed_distances, method='ward')

#     # -------------------
#     # STEP 4: Evaluate cluster solutions
#     # -------------------
#     # We'll store results for each k: silhouette, dunn, connectivity
#     cluster_evaluations = []
#     for k in range(2, max_clusters + 1):
#         labels = fcluster(Z, t=k, criterion='maxclust')  # cluster labels for each city

#         # Silhouette Score (higher is better)
#         silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')

#         # Dunn Index (higher is better)
#         dunn = dunn_index(distance_matrix, labels)

#         # Connectivity (lower is better)
#         conn = connectivity(distance_matrix, labels, neighborhood_size=10)

#         cluster_evaluations.append({
#             'k': k,
#             'silhouette': silhouette,
#             'dunn_index': dunn,
#             'connectivity': conn
#         })

#     # Decide how to select the best number of clusters
#     # For demonstration, let's pick the best k by silhouette:
#     best_k_entry = max(cluster_evaluations, key=lambda x: x['silhouette'])
#     best_k = best_k_entry['k']
#     print("\nCluster evaluations:")
#     for entry in cluster_evaluations:
#         print(entry)
#     print(f"\nBest k by silhouette = {best_k}\n")

#     # -------------------
#     # STEP 5: Final Clustering & Save
#     # -------------------
#     final_labels = fcluster(Z, t=best_k, criterion='maxclust')

#     results_df = pd.DataFrame({
#         "City": [name.replace("_metrics", "") for name in city_names],
#         "Cluster": final_labels
#     })
#     results_df.to_parquet(output_file, index=False)
#     print(f"Clustering results saved to {output_file}")


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

    # Load the cluster information
    clusters_df = pd.read_parquet(clusters_path)
    clusters = clusters_df.set_index("City")["Cluster"]

    # Initialize a dictionary to hold aggregated data for each cluster
    cluster_aggregates = {}

    print("Processing metrics for each city...")
    for city_file in os.listdir(metrics_path):
        if city_file.endswith(".parquet"):
            city_name = os.path.splitext(city_file)[0].replace("_metrics", "")

            # Check if the city exists in the cluster assignments
            if city_name not in clusters:
                print(f"Warning: {city_name} is not in the cluster assignments. Skipping.")
                continue

            # Get the cluster assignment for the city
            cluster_id = clusters[city_name]

            # Load the city's metrics
            city_metrics_path = os.path.join(metrics_path, city_file)
            city_df = pd.read_parquet(city_metrics_path)
            
            # Ensure datetime columns are processed correctly
            city_df.columns = pd.to_datetime(city_df.columns, format="%d/%m/%Y")
            city_df = city_df.T  # Transpose to make datetime the index

            # --- Select only the requested metric ---
            # NOTE: The 'metric' name must match exactly the column name in your data
            # e.g. "Percentage of Posts with Comments"
            # If your data has a slightly different naming scheme, adjust accordingly.
            if metric not in city_df.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in file columns. Available metrics: {list(city_df.columns)}"
                )
            
            city_df = city_df[[metric]]  # Keep only the chosen metric as a single column DataFrame

            # Add the city's metrics to the corresponding cluster
            if cluster_id not in cluster_aggregates:
                cluster_aggregates[cluster_id] = []
            cluster_aggregates[cluster_id].append(city_df)

    # Aggregate the metrics for each cluster
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

    # Create a final DataFrame with all cluster results
    final_df = pd.DataFrame(aggregated_results)

    # Save the aggregated metrics to a parquet file
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
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Set the '_index_level_0' column as the DataFrame index if it's present
    if '_index_level_0' in df.columns:
        df = df.set_index('_index_level_0')

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the time series for both clusters
    plt.plot(df.index, df['Cluster 0'], label='Cluster 0', color='blue')
    plt.plot(df.index, df['Cluster 1'], label='Cluster 1', color='orange')

    # Formatting
    plt.title('Aggregated Percentage of Posts Receiving Comments based on Clusters')
    plt.xlabel('Time')
    plt.ylabel('Percentage of posts receiving comments')
    plt.legend()
    plt.grid(True)

    # If a save path is provided, save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Display the plot
    plt.show()


# def predict_and_plot_cluster_aggregates(aggregated_data_path, output_plot_path):
#     """
#     Trains a time series model on the first 3 months of pre-COVID data,
#     validates it on the next 3 months, and then applies it over the COVID period (Jan 1, 2020 - Dec 31, 2021).
#     Plots predictions against actual observed aggregate scores.

#     Parameters:
#     - aggregated_data_path (str): Path to the aggregated cluster metrics parquet file.
#     - output_plot_path (str): Path to save the prediction and actual comparison plot.
#     """
#     # Load the aggregated cluster data
#     data = pd.read_parquet(aggregated_data_path)

#     # Ensure datetime index
#     data.index = pd.to_datetime(data.index)

#     # Define pre-COVID and COVID periods
#     precovid_start = data.index.min()
#     precovid_mid = precovid_start + pd.Timedelta(days=90)
#     precovid_end = precovid_start + pd.Timedelta(days=180)
#     covid_start = pd.Timestamp("2020-01-01")
#     covid_end = pd.Timestamp("2021-12-31")

#     # Train and predict for each cluster
#     plt.figure(figsize=(12, 6))
#     for cluster in data.columns:
#         print(f"Processing {cluster}...")

#         # Prepare data for Prophet
#         cluster_data = data[[cluster]].reset_index()
#         cluster_data.columns = ["ds", "y"]  # Prophet expects columns 'ds' (date) and 'y' (value)

#         # Split into training (first 3 months) and testing (second 3 months)
#         train_data = cluster_data[cluster_data["ds"] < precovid_mid]
#         test_data = cluster_data[(cluster_data["ds"] >= precovid_mid) & (cluster_data["ds"] < precovid_end)]

#         # Train the Prophet model
#         model = Prophet(weekly_seasonality=True)
#         model.fit(train_data)

#         # Validate on the test data
#         test_future = model.make_future_dataframe(periods=len(test_data), freq="W")
#         test_forecast = model.predict(test_future)

#         # Calculate error metrics for validation
#         test_predicted = test_forecast[test_forecast["ds"].isin(test_data["ds"])]["yhat"]
#         mae = mean_absolute_error(test_data["y"], test_predicted)
#         # rmse = mean_squared_error(test_data["y"], test_predicted, squared=False)
#         rmse = sqrt(mean_squared_error(test_data["y"], test_predicted))
#         print(f"Validation MAE for {cluster}: {mae:.2f}")
#         print(f"Validation RMSE for {cluster}: {rmse:.2f}")

#         # Predict over the COVID period
#         covid_future = model.make_future_dataframe(periods=len(pd.date_range(covid_start, covid_end, freq="W")), freq="W")
#         covid_forecast = model.predict(covid_future)

#         # Plot observed, validation predictions, and COVID predictions
#         plt.plot(cluster_data["ds"], cluster_data["y"], label=f"Observed {cluster}", alpha=0.6)
#         plt.plot(test_forecast["ds"], test_forecast["yhat"], linestyle="--", label=f"Validation {cluster}", alpha=0.8)
#         plt.plot(covid_forecast["ds"], covid_forecast["yhat"], linestyle="--", label=f"COVID Prediction {cluster}", alpha=0.8)

#         # Mark the split points
#         plt.axvline(precovid_mid, color="orange", linestyle="--", label="Pre-COVID Split" if cluster == data.columns[0] else "")
#         plt.axvline(covid_start, color="red", linestyle="--", label="COVID Start (2020-01-01)" if cluster == data.columns[0] else "")

#     # Add plot details
#     plt.title("Time Series Prediction During COVID Period with Pre-COVID Validation")
#     plt.xlabel("Date")
#     plt.ylabel("Aggregate Metric")
#     plt.legend()
#     plt.grid()

#     # Save and show the plot
#     plt.tight_layout()
#     plt.savefig(output_plot_path)
#     plt.show()

#     print(f"Prediction plot saved to {output_plot_path}")


def predict_and_plot_cluster_aggregates(aggregated_data_path, output_plot_path):
    """
    Trains a time series model on all data before January 1, 2020,
    and plots predictions against actual observed aggregate scores from January 1, 2020 onwards.

    Parameters:
    - aggregated_data_path (str): Path to the aggregated cluster metrics parquet file.
    - output_plot_path (str): Path to save the prediction and actual comparison plot.
    """
    # Load the aggregated cluster data
    data = pd.read_parquet(aggregated_data_path)

    # Ensure datetime index
    data.index = pd.to_datetime(data.index)

    # Define the split point for training and forecasting
    split_date = pd.Timestamp("2020-01-01")

    # Train and predict for each cluster
    plt.figure(figsize=(12, 6))
    for cluster in data.columns:
        print(f"Processing {cluster}...")

        # Prepare data for Prophet
        cluster_data = data[[cluster]].reset_index()
        cluster_data.columns = ["ds", "y"]  # Prophet expects columns 'ds' (date) and 'y' (value)

        # Split into training (before 2020) and testing (2020 onwards)
        train_data = cluster_data[cluster_data["ds"] < split_date]
        test_data = cluster_data[cluster_data["ds"] >= split_date]

        # Train the Prophet model on data before 2020
        model = Prophet(weekly_seasonality=True)
        model.fit(train_data)

        # Predict for the period starting from 2020
        future = model.make_future_dataframe(periods=len(test_data), freq="W")
        forecast = model.predict(future)

        # Calculate error metrics for the forecast
        predicted = forecast[forecast["ds"].isin(test_data["ds"])]["yhat"]
        mae = mean_absolute_error(test_data["y"], predicted)
        rmse = sqrt(mean_squared_error(test_data["y"], predicted))
        print(f"Forecast MAE for {cluster}: {mae:.2f}")
        print(f"Forecast RMSE for {cluster}: {rmse:.2f}")

        # Plot observed data and predictions
        plt.plot(cluster_data["ds"], cluster_data["y"], label=f"Observed {cluster}", alpha=0.6)
        plt.plot(forecast["ds"], forecast["yhat"], linestyle="--", label=f"Predicted {cluster}", alpha=0.8)

        # Mark the split point (January 1, 2020)
        plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01" if cluster == data.columns[0] else "")

    # Add plot details
    plt.title("Time Series Prediction from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel("Aggregate Metric")
    plt.legend()
    plt.grid()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

    print(f"Prediction plot saved to {output_plot_path}")


# def predict_and_plot_time_series(data_path, metric_name, output_plot_path):
#     """
#     Fits a Prophet model on data before January 1, 2020, for a specified metric,
#     and plots the forecast against actual data from January 1, 2020, onwards.

#     Parameters:
#     - data_path (str): Path to the parquet file containing the time series data.
#     - metric_name (str): Name of the time series metric to analyze.
#     - output_plot_path (str): Path to save the prediction and actual comparison plot.
#     """
#     # Load the data
#     data = pd.read_parquet(data_path)

#     # Ensure the date column is in datetime format
#     data['date'] = pd.to_datetime(data['date'])

#     # Filter the data for the specified metric
#     metric_data = data[data['metric'] == metric_name]

#     # Define the split point for training and forecasting
#     split_date = pd.Timestamp("2020-01-01")

#     # Split into training (before 2020) and testing (2020 onwards)
#     train_data = metric_data[metric_data['date'] < split_date]
#     test_data = metric_data[metric_data['date'] >= split_date]

#     # Prepare data for Prophet
#     train_data = train_data.rename(columns={'date': 'ds', 'value': 'y'})
#     test_data = test_data.rename(columns={'date': 'ds', 'value': 'y'})

#     # Train the Prophet model on data before 2020
#     model = Prophet(weekly_seasonality=True)
#     model.fit(train_data)

#     # Predict for the period starting from 2020
#     future = model.make_future_dataframe(periods=len(test_data), freq="D")  # Daily frequency
#     forecast = model.predict(future)

#     # Filter the forecast to only include dates from 2020 onwards
#     forecast = forecast[forecast['ds'] >= split_date]

#     # Calculate error metrics for the forecast
#     predicted = forecast['yhat']
#     mae = mean_absolute_error(test_data['y'], predicted)
#     rmse = sqrt(mean_squared_error(test_data['y'], predicted))
#     print(f"Forecast MAE for {metric_name}: {mae:.2f}")
#     print(f"Forecast RMSE for {metric_name}: {rmse:.2f}")

#     # Plot observed data and predictions
#     plt.figure(figsize=(12, 6))
#     plt.plot(metric_data['date'], metric_data['value'], label=f"Observed {metric_name}", alpha=0.6)
#     plt.plot(forecast['ds'], forecast['yhat'], linestyle="--", label=f"Predicted {metric_name}", alpha=0.8)

#     # Mark the split point (January 1, 2020)
#     plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01")

#     # Add plot details
#     plt.title(f"Time Series Prediction for {metric_name} from 2020 Onwards")
#     plt.xlabel("Date")
#     plt.ylabel("Metric Value")
#     plt.legend()
#     plt.grid()

#     # Save and show the plot
#     plt.tight_layout()
#     plt.savefig(output_plot_path)
#     plt.show()

#     print(f"Prediction plot saved to {output_plot_path}")


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

    # 1) Load the data
    data = pd.read_parquet(data_path)

    # 2) Convert columns (originally date strings) to datetime objects
    #    Adjust the date format as needed: if it's day-first, do dayfirst=True, etc.
    data.columns = pd.to_datetime(data.columns, format="%d/%m/%Y")

    # 3) Transpose so the index is the date, and columns are metrics
    data = data.T  # Now each row is a date, each column is a metric

    # 4) Make sure the requested metric exists
    if metric_name not in data.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found. "
            f"Available metrics: {list(data.columns)}"
        )

    # 5) Keep only the single requested metric
    #    This makes 'data' into a 1-column DataFrame (index=dates, column=metric_name)
    data = data[[metric_name]]

    # 6) Rename the column to 'y' and move the index to a 'ds' column for Prophet
    data.rename(columns={metric_name: 'y'}, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'ds'}, inplace=True)

    # At this point, 'data' has columns ['ds', 'y'] with ds = datetime, y = metric value

    # Define the split point for training vs. testing
    split_date = pd.Timestamp("2020-01-01")
    train_data = data[data['ds'] < split_date].copy()
    test_data  = data[data['ds'] >= split_date].copy()

    # Train a Prophet model on data prior to Jan 1, 2020
    model = Prophet(weekly_seasonality=True)
    model.fit(train_data)

    # Predict for the length of the test set
    future = model.make_future_dataframe(periods=len(test_data), freq="W")  # weekly frequency
    forecast = model.predict(future)

    # Filter forecast to only include dates from 2020 onwards
    forecast_2020plus = forecast[forecast['ds'] >= split_date]

    # Calculate error metrics
    y_true = test_data['y'].values
    y_pred = forecast_2020plus['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print(f"Forecast MAE for {metric_name}: {mae:.2f}")
    print(f"Forecast RMSE for {metric_name}: {rmse:.2f}")

    # Plot: observed data vs. predicted
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label=f"Observed {metric_name}", alpha=0.6)
    plt.plot(forecast_2020plus['ds'], forecast_2020plus['yhat'],
             linestyle="--", label=f"Predicted {metric_name}", alpha=0.8)

    # Mark the split point
    plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01")

    # Add labels and legend
    plt.title(f"Time Series Prediction for {metric_name} from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show the plot
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
    # Train the Prophet model
    model = Prophet(weekly_seasonality=True)
    model.fit(train_df)

    # Determine how many days we need to forecast to cover the test set
    last_train_date = train_df['ds'].max()
    last_test_date = test_df['ds'].max()

    # Number of days between last train date and last test date
    horizon_days = (last_test_date - last_train_date).days

    # Create a future dataframe that extends horizon_days beyond the last train date
    future = model.make_future_dataframe(periods=horizon_days, freq=freq)
    forecast = model.predict(future)

    # Evaluate only on the test portion
    # (test_df['ds'] should match some subset of forecast['ds'])
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
    # 1) Load the data
    data = pd.read_parquet(aggregated_data_path)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()  # just in case
    
    # Convert date string bounds to Timestamp
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts   = pd.Timestamp(train_end)
    test_start_ts  = pd.Timestamp(test_start)
    test_end_ts    = pd.Timestamp(test_end)
    post_test_ts   = pd.Timestamp(post_test_start)

    # We will iterate over clusters (columns)
    clusters = data.columns
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]  # make it a list so we can iterate

    # For storing metrics
    metrics = {
        'cluster': [],
        'MAE': [],
        'RMSE': []
    }

    for i, cluster in enumerate(clusters):
        ax = axes[i]

        # Create a Prophet-friendly DataFrame for this cluster
        cluster_df = data[[cluster]].reset_index()
        cluster_df.columns = ['ds', 'y']  # Prophet expects these names

        # 2) Slice the single DataFrame into train and test
        train_mask = (cluster_df['ds'] >= train_start_ts) & (cluster_df['ds'] <= train_end_ts)
        test_mask =  (cluster_df['ds'] >= test_start_ts)  & (cluster_df['ds'] <= test_end_ts)

        train_data = cluster_df.loc[train_mask]
        test_data  = cluster_df.loc[test_mask]

        # 3) Train & validate
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

        # 4) Final forecast from the end of the test period onward
        #    We'll forecast up to the maximum date in the dataset (or further).
        last_test_date = test_end_ts
        max_date = cluster_df['ds'].max()
        days_to_forecast = (max_date - last_test_date).days

        if days_to_forecast > 0:
            future_df = model.make_future_dataframe(periods=days_to_forecast, freq=freq)
            final_forecast = model.predict(future_df)
        else:
            # If there's no post-test period, skip
            final_forecast = in_sample_forecast

        # 5) Plot everything
        #   - Observed data for entire range
        ax.plot(cluster_df['ds'], cluster_df['y'], label='Observed', color='black', alpha=0.6)

        #   - In-sample forecast (through end of test period)
        ax.plot(in_sample_forecast['ds'], in_sample_forecast['yhat'], 
                color='blue', linestyle='--', label='Train+Test Prediction')

        #   - Post-test forecast
        ax.plot(final_forecast['ds'], final_forecast['yhat'], 
                color='red', linestyle='--', label='Post-Test Prediction')

        # Vertical lines to mark boundaries
        ax.axvline(train_end_ts, color='orange', linestyle='--', label='Train End')
        ax.axvline(test_end_ts,  color='green',  linestyle='--', label='Test End')

        # Zoom out a bit if desired
        ax.set_xlim([train_start_ts - pd.Timedelta(days=5), max_date + pd.Timedelta(days=5)])

        ax.set_title(f"Cluster: {cluster}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Metric")
        ax.grid(True)
        ax.legend()

    # Wrap up
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_plot_path}")

    # Show metrics
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

    # Dictionary mapping U.S. cities to their 2020 Census population sizes
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

    # df = pd.read_parquet("../comments_percentage_subclusters.parquet")
    # cluster = [df["City"][i] for i,j in enumerate(df["Cluster"]) if j ==0]
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

    # save_timeseries_metrics_for_cities(city_dict)

    # cluster_cities_with_dtw_hierarchical(
    #     city_files_folder="../metrics",
    #     output_file="../city_clusters2.parquet",
    #     resample_unit="W",
    #     max_clusters=10,
    # )

    # plot_two_clusters_timeseries("../aggregated_cluster_metrics.parquet", "../lifespan_clusters.png")

    # predict_and_plot_single_df(
    #     aggregated_data_path="../aggregated_cluster_metrics.parquet",
    #     output_plot_path="../cluster_predictions_with_validation.png"
    # )

    # predict_and_plot_cluster_aggregates(
    #     aggregated_data_path="../aggregated_response_times.parquet",
    #     output_plot_path="../response_cluster_forecast.png"
    # )

    # predict_and_plot_time_series("../metrics/losangeles_metrics.parquet", "Average Response Time (Minutes)", "../losangeles_response_prophet.png")

    # cluster_cities_with_dtw_pyclustering(
    #     city_files_folder="../metrics",
    #     output_file="../comments_percentage_subclusters.parquet",
    #     metric_name="Percentage of Posts with Comments",
    #     resample_unit="W",
    #     max_clusters=10,
    #     n_jobs=-1,
    #     selected_cities = comment_percentage_cluster_0
    # )

    cluster_cities_with_dtw_pyclustering(
        city_files_folder="../metrics",
        output_file="../response_cutoff_clusters.parquet",
        # metric_name="Average Response Time with Cutoff (Minutes)",
        metric_name="Average Response Time (Minutes)",
        resample_unit="W",
        max_clusters=10,
        n_jobs=-1,
        selected_cities=None
    )

    # aggregate_metrics_by_cluster(
    #     metrics_path="../metrics",
    #     clusters_path="../comments_percentage_clusters.parquet",
    #     output_path="../aggregated_comment_percentage.parquet",
    #     metric="Percentage of Posts with Comments",
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

    # cluster_cities_with_hierarchical_dtw_parallel(
    #     city_files_folder="../metrics",
    #     output_file="../city_clusters2.parquet",
    #     resample_unit='W',
    #     max_clusters=10,
    #     linkage_method='complete'
    # )

    # save_comment_percentage_metrics(
    #     nyc_submissions_path,
    #     nyc_comments_path,
    #     nyc_precovid_submissions_path,
    #     nyc_precovid_comments_path,
    #     city="newyorkcity",
    #     resample_unit='W',  # 'W' for weekly, 'D' for daily, 'M' for monthly, etc.
    #     output_folder='metrics'
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

    # apply_liwc_and_compute_avg_with_dict("../covid_data_parquet/newyorkcity_comments.parquet", "../covid_data_parquet/newyorkcity_submissions.parquet", "../liwc/LIWC2007_English080730.dic", "../liwc/LIWC2007_Categories.txt")

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

    # plot_post_lifespan(
    #     "../covid_data_parquet/newyorkcity_submissions.parquet", 
    #     "../covid_data_parquet/newyorkcity_comments.parquet", 
    #     "hours", 
    #     "New York City Posts Lifespan Distribution", 
    #     "Log Lifespan (Hours)", 
    #     "Number of Posts",
    #     True,
    #     (True, False),
    #     True
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