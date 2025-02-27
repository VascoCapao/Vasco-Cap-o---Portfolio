# --------------------------------------
# All imports for all notebooks
# --------------------------------------

# Standard Libraries
import os
import time
import random
import logging
import re
from collections import defaultdict

# Data Manipulation
import pandas as pd
import numpy as np
from sklearn import metrics

# Preprocessing
import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Vectorization and Clustering
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans

# Classification and Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Metrics and Evaluation
from sklearn.metrics import (
    f1_score, precision_score, recall_score, make_scorer,
    mean_squared_error, mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score
)
from scipy.stats import pearsonr

# Custom Utilities
from utils import pipeline_project

# Web Scraping
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import networkx as nx

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# Data Understanding and Preparation Functions
# ---------------------------------------------

def Nº_followers(s):
    """
    Extracts the last sequence of digits at the end of the input string.

    Parameters:
        s: The input string or object to extract digits from.

    Returns:
        str: The last sequence of digits found at the end of the string.
             Returns an empty string if no digits are found.
    """
    result = re.search(r'\d+\s*$', str(s))  
    return result.group(0) if result else ''

def Nº_reviews(s):
    """
    Extracts the first sequence of digits at the beginning of the input string.

    Parameters:
        s: The input string or object to extract digits from.

    Returns:
        str: The first sequence of digits found at the start of the string.
             Returns an empty string if no digits are found.
    """
    result =  re.search(r'^\d+', str(s))
    return result.group(0) if result else ''

def only_numbers(s):
    """
    Extracts sequences of digits from the input and returns them as a space-separated string.

    Parameters:
        s: The input string or object to extract digits from.

    Returns:
        str: A string containing sequences of digits, separated by spaces.
    """
    return ' '.join(re.findall(r'\d+', str(s)))

def display_unique_values(df):
    """
    Display the first 10 unique values for each column in the given DataFrame.

    Parameters:
    df: The DataFrame to analyze.
    """
    for column in df.columns:
        print(f"Column: {column}")
        unique_values = df[column].unique()[:10] 
        print(f"First 10 unique values: {unique_values}\n")

def plotter_1(dataset, column):
    """
    Plots the distribution of a numerical column from a dataset using two histograms: 
    one with a linear scale and the other with a logarithmic scale.

    Parameters:
        dataset: The input dataset containing the column to plot.
        column: The name of the numerical column to visualize.

    Returns:
        None: Displays two side-by-side histograms (linear and log scales) with mean and median lines.
    """
    # Set the style
    sns.set_theme(style="whitegrid")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Cap the number of bins to a reasonable maximum (e.g., 50) to avoid overcrowding
    bins = min(len(dataset[column].unique()), 50)

    # Calculate mean and median
    mean_val = dataset[column].mean()
    median_val = dataset[column].median()

    # --- Left Plot: Linear Scale ---
    axes[0].hist(dataset[column], bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[0].set_title(f"Distribution of {column} (Linear Scale)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()

    # --- Right Plot: Logarithmic Scale ---
    axes[1].hist(dataset[column], bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[1].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[1].set_yscale('log')  # Apply log scale to the y-axis
    axes[1].set_title(f"Distribution of {column} (Log Scale)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(column, fontsize=12)
    axes[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def Ngramsvectorizer(vectorizer_instance,corpus):
    """
    Applies an n-gram vectorizer to a text corpus and returns the resulting matrix, 
    list of features (n-grams), and a DataFrame summarizing word frequencies.

    Parameters:
        vectorizer_instance: An instance of a vectorizer, such as CountVectorizer or TfidfVectorizer, configured for n-grams.
        corpus: A list of text documents to vectorize.

    Returns:
        tuple:
            - matrix: A 2D array where rows represent documents and columns represent the frequency or importance of n-grams.
            - word_list: A list of n-grams corresponding to the columns in the matrix.
            - word_count_df: A DataFrame summarizing the word frequencies or importance scores, as computed by
              `pipeline_project.word_freq_calculator`.
    """
    matrix= vectorizer_instance.fit_transform(corpus).toarray()
    word_list = vectorizer_instance.get_feature_names_out().tolist()
    word_count_df = pipeline_project.word_freq_calculator(matrix,
                                         word_list,
                                         df_output= True)
    return (matrix,word_list,word_count_df)

class SimpleGroupedColorFunc:
    """
    A class for assigning colors to words in a WordCloud based on predefined groups.

    Attributes:
        color_to_words (dict): A mapping of colors to lists of words.
        default_color (str): The default color to use for words not in any group.
        word_to_color (dict): A mapping of individual words to their corresponding color.

    Methods:
        __init__(color_to_words, default_color):
            Initializes the color function with the provided mappings and default color.

        __call__(word, **kwargs):
            Returns the color for a given word, falling back to the default color if the word
            is not explicitly mapped to a group.
    """

    def __init__(self, color_to_words, default_color):
        """
        Initialize the SimpleGroupedColorFunc with color-to-word mappings and a default color.

        Args:
            color_to_words (dict): A dictionary where the keys are colors (str)
                                   and the values are lists of words (list of str).
            default_color (str): A hex or named color string to be used as the fallback color.
        """
        self.color_to_words = color_to_words
        self.default_color = default_color
        # Create a mapping of each word to its corresponding color
        self.word_to_color = {
            word: color
            for color, words in color_to_words.items()
            for word in words
        }

    def __call__(self, word, **kwargs):
        """
        Return the color for the given word.

        Args:
            word (str): The word for which the color is being requested.
            **kwargs: Additional arguments passed by the WordCloud library, ignored here.

        Returns:
            str: The color associated with the word, or the default color if not mapped.
        """
        return self.word_to_color.get(word, self.default_color)


# -----------------------------------
# Multilabel Classification Functions
# -----------------------------------

def plotter_3(dataset, title):
    """
    Creates a horizontal bar plot showing the frequency of values in each column of the dataset (excluding the first column).
    
    Args:
        dataset: The dataset containing numeric values for plotting.
        title: The title of the plot.
    
    Returns:
        None: Displays the bar plot.
    """
    # Exclude the first column (assuming it's not needed for the plot)
    numeric_columns = dataset.iloc[:, 1:]

    # Calculate the sum of values (1s) in each column and sort them in descending order
    column_sums = numeric_columns.sum(axis=0).sort_values(ascending=False)

    # Set a modern style for the plot
    sns.set_style("whitegrid")

    # Generate a gradient color palette based on the column sums
    norm = plt.Normalize(vmin=column_sums.min(), vmax=column_sums.max())  # Normalize the values
    cmap = plt.get_cmap("coolwarm")  # Choose a continuous colormap like "coolwarm"
    colors = [cmap(norm(value)) for value in column_sums.values]

    # Create the bar plot
    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    sns.barplot(x=column_sums.values, y=column_sums.index, orient="h", ax=axes, palette=colors)

    # Annotate each bar with its frequency value
    for i, value in enumerate(column_sums.values):
        axes.text(value, i, f'{int(value)}', va='center', ha='left', fontsize=12, color='black', fontweight='bold')

    # Customize the plot appearance
    axes.set_title(title, fontsize=20, fontweight='bold', color="#333333")
    axes.set_xlabel("Labels Frequency", fontsize=14, color="#555555")
    axes.set_ylabel("Columns", fontsize=14, color="#555555")
    axes.tick_params(axis='both', which='major', labelsize=12, colors="#555555")

    # Add gridlines for the x-axis
    axes.xaxis.grid(True, linestyle='--', color='gray', alpha=0.6)
    axes.yaxis.grid(False)  # Disable gridlines for the y-axis

    # Adjust the layout to prevent overlapping elements
    plt.tight_layout()

    # Display the plot
    plt.show()

def join_multi_word_phrases(column):
    """
    Converts multi-word phrases in a DataFrame column into single tokens by removing spaces between words.

    Args:
        column: A pandas Series containing text data with multi-word phrases.

    Returns:
        pd.Series: A pandas Series with multi-word phrases converted into single tokens.
    """
    # Replace spaces between words in phrases with no spaces
    return column.apply(lambda x: re.sub(r'\b(\w+)\s+(\w+)\b', r'\1\2', x))

def filter_cuisines(cuisines, invalid_labels_set):
    """
    Filters out invalid cuisines from a string of cuisines and returns a cleaned string of valid cuisines.

    Args:
        cuisines: A comma-separated string of cuisine names.
        invalid_labels_set: A set of invalid cuisine labels to exclude.

    Returns:
        str: A comma-separated string of valid cuisine names.
    """
    # Split the cuisines by comma, strip whitespace, exclude invalid ones, and rejoin as a string
    return ", ".join([cuisine.strip() for cuisine in cuisines.split(",") if f"{cuisine.strip()}_label" not in invalid_labels_set])

def fold_score_calculator(y_pred, y_test, verbose=False):
    """
    Calculates binary classification scores (accuracy, precision, recall, F1 score) for a prediction fold.

    Args:
        y_pred: The predicted labels.
        y_test: The actual labels.
        verbose: If True, prints the calculated scores. Defaults to False.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, and F1 score.
    """
    # Compute accuracy, precision, recall, and F1 score using weighted average
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    # Print the scores if verbose is True
    if verbose:
        print(f"Accuracy: {acc:.4f} \nPrecision: {prec:.4f} \nRecall: {recall:.4f} \nF1: {f1:.4f}")

    # Return the calculated scores
    return acc, prec, recall, f1

class HermeticClassifier(ClassifierMixin, BaseEstimator):
    """
    Custom classifier that integrates preprocessing, vectorization, and a final classifier.
    It supports multiple vectorization methods (e.g., TF-IDF, Word2Vec) and can handle text data for multilabel classification.
    
    Attributes:
        preprocessor: An object that handles text preprocessing.
        vectorizer: The vectorizer (e.g., TfidfVectorizer, Word2Vec).
        classifier: The machine learning model used for classification (e.g., RandomForestClassifier).
        d2v_vector_size: The vector size used in Doc2Vec model (default is 300).
        d2v_window: The window size used in Doc2Vec model (default is 6).
    """

    def __init__(self, preprocessor, vectorizer, classifier, d2v_vector_size=300, d2v_window=6, **kwargs):
        """
        Initializes the HermeticClassifier with preprocessing, vectorization, and classification components.

        Args:
            preprocessor: Preprocessing class or function.
            vectorizer: Vectorizer class or function (e.g., TfidfVectorizer, Word2Vec).
            classifier: Classifier to use for model fitting (e.g., RandomForestClassifier).
            d2v_vector_size: Size of the Doc2Vec vectors (default is 300).
            d2v_window: Window size for Doc2Vec (default is 6).
            **kwargs: Additional keyword arguments passed to the preprocessor or classifier.
        """
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.d2v_vector_size = d2v_vector_size
        self.d2v_window = d2v_window

    def fit(self, X, y, **kwargs):
        """
        Fits the HermeticClassifier to the training data.

        Args:
            X: Training data (text data).
            y: Target labels.
            **kwargs: Additional keyword arguments passed to the preprocessing or vectorization steps.

        Returns:
            self: Fitted classifier.
        """
        # Preprocess the data
        X_preproc = [self.preprocessor.main_pipeline(doc, **kwargs) or "" for doc in X]

        # Try transforming with the vectorizer
        try:
            X_train = self.vectorizer.fit_transform(X_preproc)
        except AttributeError:
            # If vectorizer is not transformable, use Doc2Vec model instead
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
            self.d2v_model = self.vectorizer(documents, vector_size=self.d2v_vector_size, window=self.d2v_window, min_count=1, workers=4)
            X_train = [self.d2v_model.dv[idx].tolist() for idx in range(len(X_preproc))]

        y_train = y
        
        # Ensure the format is correct
        try:
            X_train = X_train.toarray()
        except AttributeError:
            pass
        try:
            y_train = y_train.to_numpy()
        except AttributeError:
            pass

        # Fit the classifier
        self.classifier.fit(X_train, y_train)

        # Set the classes_ attribute
        self.classes_ = getattr(self.classifier, 'classes_', None)
        if self.classes_ is None:
            self.classes_ = np.unique(y)

        self.X_ = X_train
        self.y_ = y_train

        return self

    def predict(self, X_test_raw, **kwargs):
        """
        Makes predictions on the test data using the trained HermeticClassifier.

        Args:
            X_test_raw: Test data (text data).
            **kwargs: Additional keyword arguments passed to the preprocessing step.

        Returns:
            y_pred (array): Predicted labels for the test data.
        """
        # Ensure the model is fitted
        check_is_fitted(self)

        # Preprocess the test data
        X_test = [self.preprocessor.main_pipeline(doc, **kwargs) for doc in X_test_raw]

        # Try transforming with the vectorizer
        try:
            X_test = self.vectorizer.transform(X_test)
        except AttributeError:
            # If vectorizer is not transformable, use Doc2Vec model instead
            X_test = [self.d2v_model.infer_vector(word_tokenize(content)).tolist() for content in X_test]

        # Ensure the format is correct
        try:
            X_test = check_array(X_test.toarray())
        except AttributeError:
            X_test = check_array(X_test)
        
        # Predict using the classifier
        y_pred = self.classifier.predict(X_test)

        return y_pred

# ----------------------------
# Sentiment Analysis Functions
# ----------------------------

# Initialize the Vader sentiment analyzer
vader = SentimentIntensityAnalyzer()

def plotter_2(data, x_col, y_col):
    """
    Create a scatter plot with a regression line to visualize the correlation between two variables.

    Parameters:
    - data: The dataset containing the data to be plotted.
    - x_col: The column name to be used for the x-axis.
    - y_col: The column name to be used for the y-axis.

    The function uses Seaborn to create a scatter plot.
    """
    
    # Set a modern style for the plot
    sns.set_theme(style="darkgrid")

    # Create the scatter plot with enhancements
    plt.figure(figsize=(20, 10))  # Set figure size
    sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=x_col,          # Color points based on the x-axis values
        palette='coolwarm', # Use the 'coolwarm' palette for color
        alpha=0.7,          # Set transparency for the points
        edgecolor='k',      # Add black edges to the points
        s=100               # Set point size
    )

    # Add a regression line
    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        scatter=False,                
        color='black',                
        line_kws={"linewidth": 2, "linestyle": "dashed"}  
    )

    # Customize the title and labels
    plt.title(f"Correlation between {x_col} and {y_col}", fontsize=24, fontweight='bold')
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=18)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()

def vader_wrapper(user_review):
    """
    Analyze the sentiment of a user review using Vader sentiment analysis.

    Parameters:
    - user_review: The list of sentences to analyze.

    Returns:
    - float: The compound sentiment score of the review.
    """
    
    # Check if the input is a list of sentences
    if type(user_review) == list:
        sent_compound_list = []
        for sentence in user_review:
            # Calculate the compound score for each sentence and add it to the list
            sent_compound_list.append(vader.polarity_scores(sentence)["compound"])
        # Calculate the mean compound score for the list
        polarity = np.array(sent_compound_list).mean()
    else:
        # Calculate the compound score for a single review
        polarity = vader.polarity_scores(user_review)["compound"]
    
    return polarity

def textblob_wrapper(user_review):
    """
    Analyze the sentiment of a user review using TextBlob sentiment analysis.

    Parameters:
    - user_review (str or list of str): The text or list of sentences to analyze.

    Returns:
    - float: The average polarity score of the review.
    """
    
    # Check if the input is a list of sentences
    if type(user_review) == list:
        sent_compound_list = []
        for sentence in user_review:
            # Calculate the polarity for each sentence and add it to the list
            sent_compound_list.append(TextBlob(sentence).sentiment.polarity)
        # Calculate the mean polarity score for the list
        polarity = np.array(sent_compound_list).mean()
    else:
        # Calculate the polarity score for a single review
        polarity = TextBlob(user_review).sentiment.polarity
    
    return polarity

# -------------------------------------
# Co-occurrence Analysis and Clustering
# -------------------------------------

def extract_dishes(review, dish_list):
    """
    Extracts dish names mentioned in a review from a predefined list of dishes.

    Args:
        review: The text of the review.
        dish_list: A list of dish names to search for.

    Returns:
        list: A list of dish names found in the review.
    """
    # Create a regex pattern for matching dish names case-insensitively
    pattern = r'\b(?:' + '|'.join(re.escape(dish) for dish in dish_list) + r')\b'
    found_dishes = re.findall(pattern, review.lower())
    return found_dishes

def clean_word_list(word_list):
    """
    Cleans a list of words by removing single-character words, specific terms, and duplicates.

    Args:
        word_list: A list of words to clean.

    Returns:
        list: A cleaned list with duplicates removed.
    """
    # Remove unwanted words and single-character items
    filtered = [
        word for word in word_list 
        if len(word) > 1 and word not in {"music", "non", "soft", "quick", "bit", "top", "star", "food", "high"}
    ]
    # Remove duplicates while keeping order
    deduplicated = list(dict.fromkeys(filtered))
    return deduplicated

def cluster_df_preproc(dataset):
    """
    Preprocesses a dataset by cleaning content and generating preprocessed columns.

    Args:
        dataset: The input dataset with a "raw_content" column.

    Returns:
        pd.DataFrame: The dataset with additional preprocessed columns.
    """
    # Remove leading whitespace before punctuation
    preceding_whitespace_pattern = r"(\s)(?=[\.\,\!\?\;\:\'])"
    dataset["raw_content"] = dataset["raw_content"].map(
        lambda content: re.sub(preceding_whitespace_pattern, "", content)
    )

    # Generate a fully preprocessed column
    full_preprocessor = pipeline_project.MainPipeline().main_pipeline
    dataset["preproc_content"] = dataset["raw_content"].map(lambda content: full_preprocessor(content))

    # Generate a column for Doc2Vec preprocessed content
    doc2vec_preprocessor = pipeline_project.MainPipeline(
        lemmatized=False, no_stopwords=False, lowercase=False
    ).main_pipeline
    dataset["doc2vec_content"] = dataset["raw_content"].map(lambda content: doc2vec_preprocessor(content))
    
    return dataset

def cluster_df_vectorizer(dataset, column, name):
    """
    Vectorizes a dataset column using BOW, TF-IDF, and Doc2Vec methods.

    Args:
        dataset: The dataset to vectorize.
        column: The column to vectorize.
        name: A name for the vectorized outputs.

    Returns:
        pd.DataFrame: The dataset with additional vector columns.
    """
    # BOW vectorization
    bow_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    dataset_bow_td_matrix = bow_vectorizer.fit_transform(dataset[column]).toarray()
    dataset[f"bow_vector_{name}"] = dataset_bow_td_matrix.tolist()

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    dataset_tfidf_td_matrix = tfidf_vectorizer.fit_transform(dataset[column]).toarray()
    dataset[f"tfidf_vector_{name}"] = dataset_tfidf_td_matrix.tolist()

    # Doc2Vec vectorization
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataset[column])]
    d2v_model = Doc2Vec(documents, vector_size=300, window=6, min_count=1, workers=4, epochs=20)
    dataset[f"doc2vec_vector_{name}"] = [d2v_model.dv[idx].tolist() for idx in tqdm(range(len(dataset)))]

    return dataset

def cooccurrence_network_generator(cooccurrence_matrix_df, n_highest_words, output=None):
    """
    Creates a co-occurrence network graph based on a co-occurrence matrix.

    Args:
        cooccurrence_matrix_df: The co-occurrence matrix.
        n_highest_words: The number of top words to include in the graph.
        output: If "return", the generated figure is returned.
    """
    filtered_df = cooccurrence_matrix_df.iloc[:n_highest_words, :n_highest_words]
    graph = nx.Graph()

    # Add nodes with sizes based on word frequency
    for word in filtered_df.columns:
        graph.add_node(word, size=filtered_df[word].sum())

    # Add weighted edges based on co-occurrence frequencies
    for word1 in filtered_df.columns:
        for word2 in filtered_df.columns:
            if word1 != word2:
                graph.add_edge(word1, word2, weight=filtered_df.loc[word1, word2])

    figure = plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(graph, k=0.5)
    edge_weights = [0.1 * graph[u][v]['weight'] for u, v in graph.edges()]
    node_sizes = [data['size'] * 2 for _, data in graph.nodes(data=True)]

    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=edge_weights)
    nx.draw_networkx_labels(graph, pos, font_weight='bold', font_size=12)
    plt.show()

    if output == "return":
        return figure

def inertia_plotter(tf_matrix, max_k=10, verbose=False):
    """
    Plots inertia values for k-means clustering to find the optimal number of clusters.
    """
    x_k_nr = []
    y_inertia = []
    for k in tqdm(range(2, max_k + 1)):
        x_k_nr.append(k)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tf_matrix)
        y_inertia.append(kmeans.inertia_)
        if verbose:
            print(f"For k = {k}, inertia = {round(kmeans.inertia_, 3)}")
    fig = px.line(x=x_k_nr, y=y_inertia, markers=True)
    fig.show()

def elbow_finder(tf_matrix, max_k=10, verbose=True):
    """
    Finds the optimal number of clusters using the elbow method.
    
    Args:
        tf_matrix: The term-frequency matrix to cluster.
        max_k: The maximum number of clusters to evaluate.
        verbose: Whether to print inertia values during computation.

    Returns:
        int: The optimal number of clusters.
    """
    y_inertia = []
    for k in tqdm(range(1, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tf_matrix)
        if verbose:
            print(f"For k = {k}, inertia = {round(kmeans.inertia_, 3)}")
        y_inertia.append(kmeans.inertia_)

    # Fit a line between first and last points to find the elbow
    x = np.array([1, max_k])
    y = np.array([y_inertia[0], y_inertia[-1]])
    coefficients = np.polyfit(x, y, 1)
    line = np.poly1d(coefficients)

    # Determine the elbow point based on maximum deviation
    a = coefficients[0]
    elbow_point = max(
        range(1, max_k + 1),
        key=lambda i: abs(y_inertia[i - 1] - line(i)) / np.sqrt(a**2 + 1)
    )
    print(f"Optimal value of k according to the elbow method: {elbow_point}")
    return elbow_point

def cluster_namer(dataset, label_column_name, nr_words=3):
    """
    Assigns descriptive names to clusters based on the most frequent words.

    Args:
        dataset: The dataset containing cluster labels and content.
        label_column_name: The column with cluster labels.
        nr_words: Number of words to use for naming each cluster.

    Returns:
        pd.DataFrame: The dataset with updated cluster labels.
    """
    labels = list(set(dataset[label_column_name]))
    corpus = []

    # Create a combined document for each cluster
    for label in labels:
        label_doc = ""
        for doc in dataset["extracted_dishes_no_chicken_str"].loc[dataset[label_column_name] == label]:        
            label_doc += " " + doc
        corpus.append(label_doc)

    # Generate TF-IDF for the cluster documents
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    label_name_list = []

    for idx, document in enumerate(corpus):
        corpus_tfidf_td_matrix = tfidf_vectorizer.fit_transform(corpus)
        corpus_tfidf_word_list = tfidf_vectorizer.get_feature_names_out()

        # Calculate word frequencies for the cluster
        label_vocabulary = pipeline_project.word_freq_calculator(
            corpus_tfidf_td_matrix[idx].toarray(),
            corpus_tfidf_word_list, 
            df_output=True
        )
        label_vocabulary = label_vocabulary.head(nr_words)

        # Generate the cluster name by concatenating top words
        label_name = "_".join(label_vocabulary["words"].iloc[:nr_words])
        label_name_list.append(label_name)

    # Map cluster labels to their names
    label_name_dict = dict(zip(labels, label_name_list))
    dataset[label_column_name] = dataset[label_column_name].map(lambda label: label_name_dict[label])

    return dataset

def plotter_3d_cluster(dataset_org, vector_column_name, cluster_label_name, write_html=False, html_name="test.html"):
    """
    Plots a 3D scatterplot of clusters based on vectorized data.

    Args:
        dataset_org: The dataset with vectorized data and cluster labels.
        vector_column_name: The column containing vectorized data.
        cluster_label_name: The column with cluster labels.
        write_html: Whether to save the plot as an HTML file.
        html_name: The name of the HTML file to save the plot.

    Returns:
        None
    """
    dataset = dataset_org.copy()
    dataset = cluster_namer(dataset, cluster_label_name)

    # Reduce dimensionality to 3D
    svd_n3 = TruncatedSVD(n_components=3)
    td_matrix = np.array([[component for component in doc] for doc in dataset[vector_column_name]])
    svd_result = svd_n3.fit_transform(td_matrix)

    # Add the SVD components to the dataset
    for component in range(3):
        col_name = f"svd_d3_x{component}"
        dataset[col_name] = svd_result[:, component].tolist()

    # Generate a 3D scatterplot
    fig = px.scatter_3d(
        dataset,
        x='svd_d3_x0',
        y='svd_d3_x1',
        z='svd_d3_x2',
        color=cluster_label_name,
        title=f"{vector_column_name}__{cluster_label_name}",
        opacity=0.7,
        hover_name="preproc_content",
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    if write_html:
        fig.write_html(html_name)
    fig.show()

def top_cuisines_by_cluster(df, cuisine_col, cluster_col, blacklist=None, normalize=False):
    """
    Identifies the top 3 cuisines for each cluster.

    Args:
        df: The dataset containing cuisine and cluster data.
        cuisine_col: Column with cuisine information.
        cluster_col: Column with cluster labels.
        blacklist: List of cuisines to exclude (optional).
        normalize: Whether to normalize cuisine counts within each cluster.

    Returns:
        pd.DataFrame: A summary with the top 3 cuisines for each cluster.
    """
    cuisines_split = df[cuisine_col].fillna("").astype(str).apply(lambda x: x.split(", "))
    results = []

    # Process each cluster
    for cluster in df[cluster_col].unique():
        cluster_data = cuisines_split[df[cluster_col] == cluster]
        cuisines = cluster_data.explode()

        # Apply blacklist if provided
        if blacklist:
            cuisines = cuisines[~cuisines.isin(blacklist)]

        # Calculate frequencies (normalized if specified)
        cluster_counts = cuisines.value_counts(normalize=normalize)
        top_cuisines = cluster_counts.head(3)

        # Store the results
        results.append({
            "Cluster": cluster,
            "Top 1 Cuisine": top_cuisines.index[0] if len(top_cuisines) > 0 else None,
            "Top 1 Count": top_cuisines.iloc[0] if len(top_cuisines) > 0 else None,
            "Top 2 Cuisine": top_cuisines.index[1] if len(top_cuisines) > 1 else None,
            "Top 2 Count": top_cuisines.iloc[1] if len(top_cuisines) > 1 else None,
            "Top 3 Cuisine": top_cuisines.index[2] if len(top_cuisines) > 2 else None,
            "Top 3 Count": top_cuisines.iloc[2] if len(top_cuisines) > 2 else None,
        })
    
    return pd.DataFrame(results)

# -------------
# Web Scraping
# -------------

def extract_until_dash(text):
    """
    Extracts the part of a string before the first hyphen or dash.

    Args:
        text: The input string.

    Returns:
        str: The portion of the string before the hyphen or dash, or the original string if no hyphen/dash is found.
    """
    match = re.match(r'^(.*?)[\-–]', text)
    if match:
        return match.group(1).strip()
    return text.strip()