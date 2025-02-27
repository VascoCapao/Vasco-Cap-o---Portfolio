import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict, Counter
from tqdm import tqdm
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.base import BaseEstimator

class MainPipeline(BaseEstimator):
    """
    A class for preprocessing text data using various cleaning, lemmatization,
    and tokenization techniques. Users can configure preprocessing steps
    using parameters.
    
    Parameters:
    - print_output: Whether to print intermediate outputs.
    - no_emojis: Remove emojis if True.
    - no_hashtags: Remove hashtags if True.
    - hashtag_retain_words: Retain words in hashtags if True.
    - no_newlines: Remove newlines if True.
    - no_urls: Remove URLs if True.
    - no_punctuation: Remove punctuation if True.
    - sentiment_punctuation: Preserve sentiment-related punctuation if True.
    - no_white_spaces: Remove extra white spaces if True.
    - no_stopwords: Remove stopwords if True.
    - custom_stopwords: List of custom stopwords to remove.
    - convert_diacritics: Convert diacritics if True.
    - lowercase: Convert text to lowercase if True.
    - lemmatized: Apply lemmatization if True.
    - list_pos: Part-of-speech tags for lemmatization.
    - pos_tags_list: Define the output type of POS tags.
    - tokenized_output: Return tokenized output if True.
    """
    def __init__(self, 
                 print_output=False, 
                 no_emojis=True, 
                 no_hashtags=True,
                 hashtag_retain_words=True,
                 no_newlines=True,
                 no_urls=True,
                 no_punctuation=True,
                 sentiment_punctuation=False,
                 no_white_spaces=True,
                 no_stopwords=True,
                 custom_stopwords=[],
                 convert_diacritics=True, 
                 lowercase=True, 
                 lemmatized=True,
                 list_pos=["n", "v", "a", "r", "s"],
                 pos_tags_list="no_pos",
                 tokenized_output=False):
        self.print_output = print_output 
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_newlines = no_newlines
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.sentiment_punctuation = sentiment_punctuation
        self.no_white_spaces = no_white_spaces
        self.no_stopwords = no_stopwords
        self.custom_stopwords = custom_stopwords
        self.convert_diacritics = convert_diacritics
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.list_pos = list_pos
        self.pos_tags_list = pos_tags_list
        self.tokenized_output = tokenized_output

    def regex_cleaner(self, raw_text):
        """
        Clean text using regular expressions based on class attributes.
        
        Args:
        - raw_text: Input raw text.
        
        Returns:
        - str: Cleaned text.
        """
        # Define regex patterns for cleaning
        newline_pattern = "(\\n)"
        hashtags_at_pattern = "([#\\@@\\u0040\\uFF20\\uFE6B])"
        hashtags_ats_and_word_pattern = "([#@]\w+)"
        emojis_pattern = "([\\u2600-\\u27FF])"
        url_pattern = "(?:\\w+:\\/{2})?(?:www)?(?:\\.)?([a-z\\d]+)(?:\\.)([a-z\\d\\.]{2,})(\\/[a-zA-Z\\/\\d]+)?"
        punctuation_pattern = "[\\u0021-\\u0026\\u0028-\\u002C\\u002E-\\u002F\\u003A-\\u003F\\u005B-\\u005F\\u007C\\u2010-\\u2028\\ufeff`]+"
        apostrophe_pattern = "'(?=[A-Z\\s])|(?<=[a-z\\.\\?\\!\\,\\s])'"
        basic_pontuation = "[\\u0022-\\u0026\\u0028-\\u002F\\u003A-\\u003B\\u005B-\\u005F\\u2010-\\u2028\\ufeff]+"

        # Remove emojis if specified
        if self.no_emojis:
            clean_text = re.sub(emojis_pattern, "", raw_text)
        else:
            clean_text = raw_text

        # Remove or process hashtags
        if self.no_hashtags:
            if self.hashtag_retain_words:
                clean_text = re.sub(hashtags_at_pattern, "", clean_text)
            else:
                clean_text = re.sub(hashtags_ats_and_word_pattern, "", clean_text)

        # Remove newlines
        if self.no_newlines:
            clean_text = re.sub(newline_pattern, " ", clean_text)

        # Remove URLs
        if self.no_urls:
            clean_text = re.sub(url_pattern, "", clean_text)

        # Remove punctuation
        if self.no_punctuation:
            clean_text = re.sub(punctuation_pattern, "", clean_text)
            clean_text = re.sub(apostrophe_pattern, "", clean_text)

        # Preserve sentiment punctuation
        if self.sentiment_punctuation:
            clean_text = re.sub(basic_pontuation, "", clean_text)

        # Remove extra white spaces
        if self.no_white_spaces:
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def lemmatize_all(self, token):
        """
        Lemmatize a token using WordNet lemmatizer.
        
        Args:
        - token: Input token.
        
        Returns:
        - str: Lemmatized token.
        """
        wordnet_lem = nltk.stem.WordNetLemmatizer()
        for arg_1 in self.list_pos:
            token = wordnet_lem.lemmatize(token, arg_1)
        return token

    def main_pipeline(self, raw_text):
        """
        Preprocess input text using the configured pipeline steps.
        
        Args:
        - raw_text: Input raw text.
        
        Returns:
        - str or list: Preprocessed text (tokenized or detokenized).
        """
        if self.print_output:
            print("Preprocessing the following input: \n>> {}".format(raw_text))

        # Clean text using regex
        clean_text = self.regex_cleaner(raw_text)

        if self.print_output:
            print("Regex cleaner returned the following: \n>> {}".format(clean_text))

        # Tokenize text
        tokenized_text = nltk.tokenize.word_tokenize(clean_text)

        # Normalize contractions
        tokenized_text = [re.sub("'m", "am", token) for token in tokenized_text]
        tokenized_text = [re.sub("n't", "not", token) for token in tokenized_text]
        tokenized_text = [re.sub("'s", "is", token) for token in tokenized_text]

        # Remove stopwords
        if self.no_stopwords:
            stopwords = nltk.corpus.stopwords.words("english")
            tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]

        # Convert diacritics
        if self.convert_diacritics:
            tokenized_text = [unidecode(token) for token in tokenized_text]

        # Lemmatize tokens
        if self.lemmatized:
            tokenized_text = [self.lemmatize_all(token) for token in tokenized_text]

        # Remove custom stopwords
        if self.no_stopwords:
            tokenized_text = [item for item in tokenized_text if item.lower() not in self.custom_stopwords]

        # POS tagging
        if self.pos_tags_list in ["pos_list", "pos_tuples", "pos_dictionary"]:
            pos_tuples = nltk.tag.pos_tag(tokenized_text)
            pos_tags = [pos[1] for pos in pos_tuples]

        # Convert to lowercase
        if self.lowercase:
            tokenized_text = [item.lower() for item in tokenized_text]

        # Return based on output configuration
        if self.pos_tags_list == "pos_list":
            return (tokenized_text, pos_tags)
        elif self.pos_tags_list == "pos_tuples":
            return pos_tuples
        else:
            if self.tokenized_output:
                return tokenized_text
            else:
                detokenizer = TreebankWordDetokenizer()
                detokens = detokenizer.detokenize(tokenized_text)
                if self.print_output:
                    print("Pipeline returning the following result: \n>> {}".format(str(detokens)))
                return str(detokens)

        
# Functions outside of the Class

def regex_cleaner(
    raw_text,
    no_emojis=True,
    no_hashtags=True,
    hashtag_retain_words=True,
    no_newlines=True,
    no_urls=True,
    no_punctuation=False,
    sentiment_punctuation=False,
    no_white_spaces=True
):
    """
    Cleans text data using various regular expressions.
    
    Parameters:
    - raw_text: Input text to be cleaned.
    - no_emojis : Remove emojis if True.
    - no_hashtags: Remove hashtags if True.
    - hashtag_retain_words: Retain hashtag words if True.
    - no_newlines: Remove newline characters if True.
    - no_urls: Remove URLs if True.
    - no_punctuation: Remove punctuation if True.
    - sentiment_punctuation: Retain sentiment punctuation.
    - no_white_spaces: Remove extra white spaces if True.

    Returns:
    - str: Cleaned text.
    """
    # Define regex patterns
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "([\u2600-\u27FF])"
    url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?"
    punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002D\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    basic_punctuation = "[\u0022-\u0026\u0028-\u002F\u003A-\u003B\u005B-\u005F\u2010-\u2028\ufeff]+"

    # Remove emojis if enabled
    if no_emojis:
        clean_text = re.sub(emojis_pattern, "", raw_text)
    else:
        clean_text = raw_text

    # Remove hashtags and mentions
    if no_hashtags:
        if hashtag_retain_words:
            clean_text = re.sub(hashtags_at_pattern, "", clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern, "", clean_text)

    # Remove newline characters
    if no_newlines:
        clean_text = re.sub(newline_pattern, " ", clean_text)

    # Remove URLs
    if no_urls:
        clean_text = re.sub(url_pattern, "", clean_text)

    # Remove punctuation
    if no_punctuation:
        clean_text = re.sub(punctuation_pattern, "", clean_text)
        clean_text = re.sub(apostrophe_pattern, "", clean_text)

    # Remove only non-sentiment punctuation
    if sentiment_punctuation:
        clean_text = re.sub(basic_punctuation, "", clean_text)

    # Remove extra white spaces
    if no_white_spaces:
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def lemmatize_all(token, list_pos=["n", "v", "a", "r", "s"]):
    """
    Lemmatizes a token for all specified parts of speech (POS).
    
    Parameters:
    - token: The word to be lemmatized.
    - list_pos: List of POS tags to use for lemmatization.

    Returns:
    - str: The lemmatized token.
    """
    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token


def main_pipeline(
    raw_text,
    print_output=True,
    no_stopwords=True,
    custom_stopwords=[],
    convert_diacritics=True,
    lowercase=True,
    lemmatized=True,
    list_pos=["n", "v", "a", "r", "s"],
    stemmed=False,
    pos_tags_list="no_pos",
    tokenized_output=False,
    **kwargs
):
    """
    Performs text preprocessing based on specified parameters.

    Parameters:
    - raw_text: The input text to preprocess.
    - print_output: Print raw and processed text if True.
    - no_stopwords: Remove stopwords if True.
    - custom_stopwords: Additional stopwords to remove.
    - convert_diacritics: Replace accented characters if True.
    - lowercase: Convert text to lowercase if True.
    - lemmatized: Lemmatize tokens if True.
    - list_pos: List of POS tags to use for lemmatization.
    - stemmed: Apply stemming if True.
    - pos_tags_list: Include POS tags ("pos_list", "pos_tuples", "pos_dictionary").
    - tokenized_output: Return tokenized text if True.
    - kwargs: Additional keyword arguments for `regex_cleaner`.

    Returns:
    - str or list: Processed text or tokenized text based on `tokenized_output`.
    """
    clean_text = regex_cleaner(raw_text, **kwargs)
    tokenized_text = nltk.tokenize.word_tokenize(clean_text)

    # Normalize contractions
    tokenized_text = [re.sub("'m", "am", token) for token in tokenized_text]
    tokenized_text = [re.sub("n't", "not", token) for token in tokenized_text]
    tokenized_text = [re.sub("'s", "is", token) for token in tokenized_text]

    # Remove stopwords
    if no_stopwords:
        stopwords = nltk.corpus.stopwords.words("english")
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]

    # Convert diacritics to standard characters
    if convert_diacritics:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    # Lemmatize tokens
    if lemmatized:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]

    # Apply stemming
    if stemmed:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]

    # Remove custom stopwords
    if no_stopwords:
        tokenized_text = [item for item in tokenized_text if item.lower() not in custom_stopwords]

    # Handle POS tags
    if pos_tags_list in ["pos_list", "pos_tuples", "pos_dictionary"]:
        pos_tuples = nltk.tag.pos_tag(tokenized_text)
        pos_tags = [pos[1] for pos in pos_tuples]

    # Convert to lowercase
    if lowercase:
        tokenized_text = [item.lower() for item in tokenized_text]

    # Output processing
    if print_output:
        print(raw_text)
        print(tokenized_text)

    if pos_tags_list == "pos_list":
        return tokenized_text, pos_tags
    elif pos_tags_list == "pos_tuples":
        return pos_tuples
    else:
        if tokenized_output:
            return tokenized_text
        else:
            detokenizer = TreebankWordDetokenizer()
            detokens = detokenizer.detokenize(tokenized_text)
            return str(detokens)

def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):
    """
    Generates a co-occurrence matrix from preprocessed sentences.
    
    Parameters:
    - preproc_sentences: Preprocessed sentences, each represented as a list of tokens.
    - sentence_cooc: If True, uses sentence-level co-occurrence. Otherwise, uses a sliding window.
    - window_size: The size of the sliding window for token co-occurrence.

    Returns:
    - pd.DataFrame: A DataFrame representing the co-occurrence matrix.
    """
    co_occurrences = defaultdict(Counter)

    # Compute co-occurrences based on sentence or window
    if sentence_cooc:
        for sentence in tqdm(preproc_sentences):
            for token_1 in sentence:
                for token_2 in sentence:
                    if token_1 != token_2:  # Avoid self-co-occurrence
                        co_occurrences[token_1][token_2] += 1
    else:
        for sentence in tqdm(preproc_sentences):
            for i, word in enumerate(sentence):
                # Consider neighboring words within the window
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:  # Avoid self-co-occurrence
                        co_occurrences[word][sentence[j]] += 1

    # Ensure that all words are unique
    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Map words to indices for matrix population
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    # Reorder rows and columns based on total co-occurrence counts
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    return co_matrix_df


def word_freq_calculator(td_matrix, word_list, df_output=True):
    """
    Calculates word frequencies from a term-document matrix.

    Parameters:
    - td_matrix: Term-document matrix.
    - word_list: List of words corresponding to the rows/columns of the matrix.
    - df_output: If True, returns a DataFrame. Otherwise, returns a dictionary.

    Returns:
    - pd.DataFrame or dict: Word frequencies sorted in descending order.
    """
    # Calculate the sum of occurrences for each word
    word_counts = np.sum(td_matrix, axis=0).tolist()
    
    if not df_output:
        # Return as a dictionary
        word_counts_dict = dict(zip(word_list, word_counts))
        return word_counts_dict
    else:
        # Return as a DataFrame for easier visualization
        word_counts_df = pd.DataFrame({"words": word_list, "frequency": word_counts})
        word_counts_df = word_counts_df.sort_values(by=["frequency"], ascending=False)
        return word_counts_df


def plot_term_frequency(df, nr_terms, df_name, show=True):
    """
    Plots the top N term frequencies from a DataFrame.

    Parameters:
    - df: DataFrame with 'words' and 'frequency' columns.
    - nr_terms: Number of top terms to display in the plot.
    - df_name: Name of the DataFrame (used in the plot title).
    - show: If True, displays the plot.

    Returns:
    - matplotlib.figure.Figure: The figure object of the plot.
    """
    # Set the Seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Create a bar plot of term frequencies
    plt.figure(figsize=(12, 8))
    sns_plot = sns.barplot(
        x='frequency', 
        y='words', 
        data=df.head(nr_terms), 
        palette=sns.color_palette("coolwarm", nr_terms)
    )
    
    # Add frequency labels to each bar
    for index, value in enumerate(df.head(nr_terms)['frequency']):
        plt.text(value, index, f'{value:.0f}', va='center', ha='left', fontsize=10, color='black')

    # Add titles and labels for better readability
    plt.title(f'Top {nr_terms} Term Frequencies of {df_name}', fontsize=20, fontweight='bold')
    plt.xlabel('Frequency', fontsize=16)
    plt.ylabel('Words', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot if specified
    if show:
        plt.show()

    # Return the figure object for further customization or saving
    fig = sns_plot.get_figure()
    plt.close()

    return fig