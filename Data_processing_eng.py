import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
import string


stop_words = set(stopwords.words('english'))
stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also',
                   'across', 'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))
set_punctuation = string.punctuation
set_punctuation = set_punctuation+'«'+'»'
stemmer = SnowballStemmer("english")


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    lemSentence = ""
    for word in sentence.split():
        lem = lemmatizer.lemmatize(word)
        lemSentence += lem
        lemSentence += " "
    lemSentence = lemSentence.strip()
    return lemSentence


def preprocessing_df_eng(df, text_column):
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].apply(cleanHtml)
    df[text_column] = df[text_column].apply(cleanPunc)
    df[text_column] = df[text_column].apply(keepAlpha)
    df[text_column] = df[text_column].apply(removeStopWords)
    df[text_column] = df[text_column].apply(lemmatize)
    return df
