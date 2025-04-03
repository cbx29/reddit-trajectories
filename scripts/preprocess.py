#!/usr/bin/env python3
import os
import json
import re
import subprocess
import pandas as pd
from datetime import datetime, timedelta

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


def combine_texts(covid_posts_path, covid_comments_path, prophet_posts_path, prophet_comments_path, output_path):
    covid_posts = pd.read_parquet(covid_posts_path)
    covid_comments = pd.read_parquet(covid_comments_path)
    precovid_posts = pd.read_parquet(prophet_posts_path)
    precovid_comments = pd.read_parquet(prophet_comments_path)

    covid_post_texts = [{'text': covid_posts['title'][i] + " " + j, 'created_utc': int(covid_posts['created_utc'][i])} for i, j in enumerate(covid_posts['selftext']) if j != '' and j != '[deleted]' and j != '[removed]']
    covid_comment_texts = [{'text': j, 'created_utc': int(covid_comments['created_utc'][i])} for i, j in enumerate(covid_comments['body']) if j != '' and j != '[deleted]' and j != '[removed]']
   
    precovid_post_texts = [{'text': precovid_posts['title'][i] + " " + j, 'created_utc': int(precovid_posts['created_utc'][i])} for i, j in enumerate(precovid_posts['selftext']) if j != '' and j != '[deleted]' and j != '[removed]']
    precovid_comment_texts = [{'text': j, 'created_utc': int(precovid_comments['created_utc'][i])} for i, j in enumerate(precovid_comments['body']) if j != '' and j != '[deleted]' and j != '[removed]']

    texts = precovid_post_texts + precovid_comment_texts + covid_post_texts + covid_comment_texts

    df = pd.DataFrame(texts)
    df.sort_values('created_utc', inplace=True)

    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    print("hello")