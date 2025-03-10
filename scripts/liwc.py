import re
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import glob
import json
import statsmodels.api as sm
from collections import Counter
import itertools

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


def get_liwc_percentage_metrics_combined_texts(
    text_file,
    liwc_dict,
    category_map,
    emotion_word,
    text_column='text',
    resample_unit='W'
):
    import pandas as pd

    def process_data(posts_path):
        posts = pd.read_parquet(posts_path, columns=['created_utc', text_column])
        posts['created_datetime'] = pd.to_datetime(posts['created_utc'], unit='s')
        posts.sort_values('created_datetime', inplace=True)

        posts['liwc_percentage'] = posts[text_column].apply(
            lambda text: analyze_emotion(text, liwc_dict, category_map, emotion_word)
        )

        posts.set_index('created_datetime', inplace=True)

        liwc_metric = posts['liwc_percentage'].resample(resample_unit).mean()
        return liwc_metric

    combined_liwc = process_data(text_file)

    full_index = pd.date_range(
        start=combined_liwc.index.min(),
        end=combined_liwc.index.max(),
        freq=resample_unit
    )
    combined_liwc = combined_liwc.reindex(full_index)

    combined_liwc = combined_liwc.interpolate()
    
    metrics_table = pd.DataFrame(
        {f'LIWC Percentage for {emotion_word}': combined_liwc.values},
        index=combined_liwc.index.strftime('%d/%m/%Y')
    ).transpose()

    return metrics_table


def save_liwc_timeseries_metrics_for_cities_combined_texts(city_dict, categories):
    output_dir = "../liwc_metrics"
    os.makedirs(output_dir, exist_ok=True)

    for city_key, city_name in city_dict.items():
        city_lower = city_key.lower().replace(" ", "")

        text_path = f"../liwc_texts/{city_lower}_texts.parquet"

        liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
        category_map = load_category_mapping(category_mapping_path)

        metrics_list = []
        for category in categories:
            print(city_lower, category)
            metric = get_liwc_percentage_metrics_combined_texts(
                text_path,
                liwc_dict,
                category_map,
                category,
                text_column='text',
                resample_unit='W'
            )
            metrics_list.append(metric)

        combined_metrics = pd.concat(metrics_list, axis=0)

        output_path = os.path.join(output_dir, f"{city_lower}_liwc_metrics.parquet")
        combined_metrics.to_parquet(output_path)


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
    baseline_end   = pd.to_datetime("2019-12-31")
    disaster_start = pd.to_datetime("2020-01-01")
    disaster_end   = pd.to_datetime("2022-12-31")
    
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
    
    df = df.loc["2016-01-01":"2021-12-31"]
    
    df_resampled = df.resample(resample_unit).mean().interpolate()
    
    smoothed_series = {}
    x = range(len(df_resampled))
    
    for metric in df_resampled.columns:
        y = df_resampled[metric].values
        # LOESS smoothing with a span (frac) of 20%
        smoothed = sm.nonparametric.lowess(y, x, frac=0.2)
        smoothed_y = smoothed[:, 1]
        smoothed_series[metric] = smoothed_y
    
    smoothed_df = pd.DataFrame(smoothed_series, index=df_resampled.index).T
    smoothed_df.columns = smoothed_df.columns.strftime("%d/%m/%Y")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    smoothed_df.to_parquet(output_path)
    
    return smoothed_df


def dtw_distance(ts_a, ts_b):
    """
    Computes the DTW distance between two 1D time series and normalizes
    the total cost by the length of the optimal warping path (i.e. returns
    the average cost per step).
    """
    n, m = len(ts_a), len(ts_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Build the DTW cost matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts_a[i - 1] - ts_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],    # insertion
                                   dtw[i, j - 1],    # deletion
                                   dtw[i - 1, j - 1])  # match

    # Backtrack to compute the length of the optimal warping path.
    i, j = n, m
    path_length = 0
    while i > 0 or j > 0:
        path_length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if dtw[i - 1, j - 1] <= dtw[i - 1, j] and dtw[i - 1, j - 1] <= dtw[i, j - 1]:
                i -= 1
                j -= 1
            elif dtw[i - 1, j] <= dtw[i, j - 1]:
                i -= 1
            else:
                j -= 1

    return dtw[n, m] / path_length


def classify_dimension_trajectories(parquet_file_path, start_date, category, end_date=None, 
                                    scale_threshold=0.5, shape_threshold=0.5):
    """
    For a given dimension represented by multiple clusters (columns in the parquet file),
    this function computes:
    
    1. Characteristic Scale:
       For each cluster, defined as the root-mean-square (RMS) magnitude of its average trajectory.
       The pairwise scale difference is the absolute difference between the RMS values of two clusters.
    
    2. Shape Difference:
       Each cluster's average trajectory is normalized by dividing by its largest absolute magnitude
       (so that its maximum magnitude becomes 1). The pairwise shape difference is then estimated as 
       the DTW distance between these normalized trajectories.
    
    The function computes these differences for all pairs of clusters and averages the results.
    If both the average scale difference and average shape difference fall below their respective
    thresholds (default: 0.5), the dimension is classified as having similar (or "universal") trajectories.
    
    Parameters:
      parquet_file_path (str): Path to the parquet file.
      start_date (str): Start date in a format parseable by pd.to_datetime.
      end_date (str, optional): End date filter.
      scale_threshold (float): Threshold for the average scale difference.
      shape_threshold (float): Threshold for the average shape difference.
    
    Returns:
      dict: Contains the averaged 'scale_difference', 'shape_difference', and 
            'classification' (either 'universal' or 'distinct').
    """
    df = pd.read_parquet(parquet_file_path)
    
    if '_index_level_0' in df.columns:
        df = df.set_index('_index_level_0')
 
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y")

    df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    clusters = df.columns.tolist()
    if len(clusters) < 2:
        raise ValueError("The data must contain at least two clusters for comparison.")
    
    scales = {}
    norm_trajectories = {}
    for col in clusters:
        traj = df[col].values
        rms = np.sqrt(np.mean(traj ** 2))
        scales[col] = rms
        max_abs = np.max(np.abs(traj))
        norm_trajectories[col] = traj if max_abs == 0 else traj / max_abs
    
    scale_diffs = []
    shape_diffs = []
    for col_a, col_b in itertools.combinations(clusters, 2):
        scale_diff_pair = abs(scales[col_a] - scales[col_b])
        shape_diff_pair = dtw_distance(norm_trajectories[col_a], norm_trajectories[col_b])
        scale_diffs.append(scale_diff_pair)
        shape_diffs.append(shape_diff_pair)
    
    avg_scale_diff = np.mean(scale_diffs)
    avg_shape_diff = np.mean(shape_diffs)
    
    classification = "universal" if (avg_scale_diff < scale_threshold and avg_shape_diff < shape_threshold) else "distinct"
    
    print("Characteristic Scales:")
    for col in clusters:
        print(f"  Cluster '{col}': RMS = {scales[col]:.3f}")
    print(f"\nAverage Scale Difference: {avg_scale_diff:.3f}")
    print(f"Average Shape Difference: {avg_shape_diff:.3f}")
    print(f"Trajectory Classification: {classification}")
    
    return {
        "category": category,
        "scale_difference": avg_scale_diff,
        "shape_difference": avg_shape_diff,
        "classification": classification
    }


def extract_emotion_tokens(text, liwc_dict, target_codes):
    """
    Tokenizes text and returns tokens that belong to any of the target LIWC codes.
    
    Parameters:
      text (str): Input text.
      liwc_dict (dict): Maps tokens to LIWC codes.
      target_codes (set): LIWC codes that correspond to the target category.
      
    Returns:
      list: Tokens that match the target category.
    """
    tokens = tokenize(text)
    matching_tokens = []
    for token in tokens:
        if token in liwc_dict:
            for code in liwc_dict[token]:
                if code in target_codes:
                    matching_tokens.append(token)
                    break  # Found a match; move on to the next token.
    return matching_tokens


def get_overall_liwc_word_shift_metrics_for_category(
    city_dict,
    liwc_dict,
    category_map,
    texts_dir,
    category,
    text_column='text'
):
    """
    Aggregates LIWC word counts across all cities for a given LIWC category and computes word-shift metrics,
    normalizing by the overall token count in each period.
    
    For each city's parquet file (with columns 'created_utc' and text_column):
      - Convert 'created_utc' (assumed to be Unix epoch seconds) to datetime.
      - Split data into pre-disaster (< Jan 1, 2021) and disaster (>= Jan 1, 2021).
      - For each text, count overall tokens and extract LIWC tokens matching the target category.
      - Aggregate token counts and overall token counts.
    
    Returns:
      A DataFrame containing the top 50 tokens (by pre-disaster LIWC counts) with:
         "words", "pre_disaster_frequency", "disaster_frequency", "percentage_change"
      where frequencies are normalized by the overall token count in each period.
    """
    pre_counter = Counter()
    dis_counter = Counter()
    pre_total_tokens = 0
    dis_total_tokens = 0

    # Use all posts before Jan 1, 2021 as pre-disaster.
    pre_disaster_end = pd.Timestamp('2021-01-01')
    disaster_start = pd.Timestamp('2021-01-01')
    
    # Precompute target LIWC codes for the given category.
    target_codes = {code for code, cat in category_map.items() if cat.lower() == category.lower()}
    
    for city_key in city_dict.keys():
        print(f"Processing city: {city_key}")
        city_lower = city_key.lower().replace(" ", "")
        file_path = os.path.join(texts_dir, f"{city_lower}_texts.parquet")
        
        try:
            df = pd.read_parquet(file_path, columns=['created_utc', text_column])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
        df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
        
        pre_df = df[df['created_datetime'] < pre_disaster_end]
        dis_df = df[df['created_datetime'] >= disaster_start]
        
        for text in pre_df[text_column].dropna():
            tokens_all = tokenize(text)
            pre_total_tokens += len(tokens_all)
            tokens = extract_emotion_tokens(text, liwc_dict, target_codes)
            pre_counter.update(tokens)
            
        for text in dis_df[text_column].dropna():
            tokens_all = tokenize(text)
            dis_total_tokens += len(tokens_all)
            tokens = extract_emotion_tokens(text, liwc_dict, target_codes)
            dis_counter.update(tokens)
    
    top_50 = pre_counter.most_common(50)
    rows = []
    for word, pre_count in top_50:
        dis_count = dis_counter.get(word, 0)
        norm_pre = pre_count / pre_total_tokens if pre_total_tokens > 0 else 0
        norm_dis = dis_count / dis_total_tokens if dis_total_tokens > 0 else 0
        # Percentage change based on normalized frequencies.
        percentage_change = ((norm_dis - norm_pre) / norm_pre * 100) if norm_pre > 0 else 0
        rows.append({
            "words": word,
            "pre_disaster_frequency": norm_pre,
            "disaster_frequency": norm_dis,
            "percentage_change": percentage_change
        })
    
    return pd.DataFrame(rows)


def save_liwc_word_shift_metrics_for_categories(
    city_dict,
    categories,
    liwc_dictionary_path,
    category_mapping_path,
    texts_dir,
    output_dir,
    text_column='text'
):
    """
    For each LIWC category in categories, aggregates normalized word-shift metrics across all cities
    and saves a separate parquet file named {category}_word_shifts.parquet in output_dir.
    
    Each output file contains:
       - "words": Top 50 tokens for the LIWC category.
       - "pre_disaster_frequency": Normalized frequency in pre-disaster period (< Jan 1, 2021).
       - "disaster_frequency": Normalized frequency in disaster period (>= Jan 1, 2021).
       - "percentage_change": Percentage change in normalized frequency.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
    category_map = load_category_mapping(category_mapping_path)
    
    for category in categories:
        print(f"Processing category: {category}")
        metrics_df = get_overall_liwc_word_shift_metrics_for_category(
            city_dict=city_dict,
            liwc_dict=liwc_dict,
            category_map=category_map,
            texts_dir=texts_dir,
            category=category,
            text_column=text_column
        )
        output_file = os.path.join(output_dir, f"{category}_word_shifts.parquet")
        metrics_df.to_parquet(output_file, index=False)
        print(f"Saved {category} word shifts to {output_file}")


if __name__ == "__main__":
    liwc_dictionary_path = '../liwc/LIWC2007_English080730.dic'
    category_mapping_path = '../liwc/LIWC2007_Categories.txt'

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
        # "affect",
        # "posemo",
        # "negemo",
        # "anx",
        # "anger",
        # "sad",
        # "swear",
        # "achieve",
        # "social",
        # "we",
        # "family",
        # "cause",
        # "tentat",
        # "certain",
        # "insight",
        "health",
        # "ingest",
        # "bio",
        # "body",
        # "motion",
        # "space",
        # "time",
        # "home",
        # "work",
        # "money"
    ]

    universal_traj_categories = [
        "anx",
        # "cause",
        "health",
        # "home",
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

    save_liwc_word_shift_metrics_for_categories(
        city_dict=city_dict,
        categories=categories,
        liwc_dictionary_path=liwc_dictionary_path,
        category_mapping_path=category_mapping_path,
        texts_dir="../liwc_texts",
        output_dir="../liwc_word_shifts",
        text_column='text'
    )
