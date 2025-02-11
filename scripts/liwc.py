import re
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import glob
import json

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


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def analyze_text(text, liwc_dict):
    tokens = tokenize(text)
    total_words = len(tokens)
    
    category_counts = defaultdict(int)
    
    for token in tokens:
        if token in liwc_dict:
            for category in liwc_dict[token]:
                category_counts[category] += 1
    
    category_percentages = {cat: (count / total_words * 100) for cat, count in category_counts.items()}
    return category_percentages


def analyze_emotions(text, liwc_dict, emotion_codes):
    tokens = tokenize(text)
    # print(tokens)
    total_words = len(tokens)
    
    emotion_counts = defaultdict(int)
    
    for token in tokens:
        if token in liwc_dict:
            # print(token, liwc_dict[token])
            for code in liwc_dict[token]:
                if code in emotion_codes.values():
                    # print(f"Token: {token} matched emotion code: {code}")
                    emotion_counts[code] += 1
    
    emotion_percentages = {code: (count / total_words * 100) for code, count in emotion_counts.items()}
 
    readable_emotions = {emotion_name: emotion_percentages.get(code, 0)
                         for emotion_name, code in emotion_codes.items()}
    return readable_emotions


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


if __name__ == '__main__':
    liwc_dictionary_path = '../liwc/LIWC2007_English080730.dic'
    category_mapping_path = '../liwc/LIWC2007_Categories.txt'
    liwc_dict = load_liwc_dictionary(liwc_dictionary_path)
    category_map = load_category_mapping(category_mapping_path)

    emotion_codes = {
        'swear': '22',
        'affect': '125',
        'posemo': '126',
        'negemo': '127',
        'anx': '128',
        'anger': '129',
        'sad': '130'
    }

    emotion_codes_2 = {
        'swear' : '22',
        'affect': '125'
    }
    
    
    sample_text = "I am very happy and excited."
    sample_text_2 = "I am feeling very happy today, but sometimes anxiety and sadness creep in unexpectedly. Overall, there is a mix of joy and a touch of worry."
    
    # liwc_results = analyze_text(sample_text, liwc_dict)
    # emotion_results = analyze_emotions(sample_text_2, liwc_dict, emotion_codes)
    
    emotion_results = analyze_emotion(sample_text_2, liwc_dict, category_map, "affect")
    print(emotion_results)

    # category_map = load_category_mapping(category_mapping_path)
    
    # print("LIWC Analysis Results:")
    # # for category, percentage in liwc_results.items():
    # #     print(f"{category}: {percentage:.2f}%")
    
    # for codes, percentage in liwc_results.items():
    #     # Split the codes and replace each with the category name using the mapping
    #     code_list = codes.split()
    #     names = [category_map.get(code, code) for code in code_list]
    #     print(f"{' '.join(names)}: {percentage:.2f}%")

    # print("Emotion Analysis Results:")
    # for emotion, percentage in emotion_results.items():
    #     print(f"{emotion}: {percentage:.2f}%")

    city_dict = {
        "albuquerque": "Albuquerque"
    }


   