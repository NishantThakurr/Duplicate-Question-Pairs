import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
import gensim
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

tfidfvectorizer = pickle.load(open("C:\\Users\\91701\\DataScience\\QuoraDuplicateQuestionpairs\\tfidfvectorizer.pkl",'rb'))


def preprocessing(q):
    q = str(q).lower().strip()
    q = BeautifulSoup(q)
    q = q.get_text()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did", "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would", "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def process_features_test(q1, q2):
    q1=preprocessing(q1)
    q2=preprocessing(q2)
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    features = [0.0] * 10
    q1_tokens = word_tokenize(q1)  # tokenization
    q2_tokens = word_tokenize(q2)

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return features
    common_token_count = len(list([tk for tk in q1_tokens if tk in q2_tokens]))

    q1_words = list([word for word in q1_tokens if word not in sw])  ##words
    q2_words = list([word for word in q2_tokens if word not in sw])

    q1_stopword = list([word for word in q1_tokens if word in sw])  ##stopwords
    q2_stopword = list([word for word in q2_tokens if word in sw])

    common_word_count = len(list([word for word in q1_words if word in q2_words]))
    common_stopword_count = len(list([stopword for stopword in q1_stopword if stopword in q2_stopword]))
    common_token_count = len(list([token for token in q1_tokens if token in q2_tokens]))

    features[0] = abs(len(q1_tokens) - len(q2_tokens))  # lendiff
    features[1] = (len(q2_tokens) + len(q1_tokens)) / 2  # meanlen
    features[2] = common_word_count / (min(len(q1_words), len(q2_words)) + 0.0001)  # mincommonwords
    features[3] = common_word_count / (max(len(q1_words), len(q2_words)) + 0.0001)  # maxcommonwords
    features[4] = common_stopword_count / (min(len(q1_stopword), len(q2_stopword)) + 0.0001)  # mincommonstopword
    features[5] = common_stopword_count / (max(len(q1_stopword), len(q2_stopword)) + 0.0001)  # maxcommonsw
    features[6] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + 0.0001)  # mintk
    features[7] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + 0.0001)  # maxtk

    features[8] = int(q1_tokens[-1] == q2_tokens[-1])  # lastwordequal
    features[9] = int(q1_tokens[0] == q2_tokens[0])  # firstwordequal

    return features


def process_fuzzy_features_test(q1, q2):
    fuzzy_features = [0.0] * 7

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # W_ratio
    fuzzy_features[1] = fuzz.WRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[2] = fuzz.partial_ratio(q1, q2)

    # partial_token_set_ratio
    fuzzy_features[3] = fuzz.partial_token_set_ratio(q1, q2)

    # partial_token_sort_ratio
    fuzzy_features[4] = fuzz.partial_token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[5] = fuzz.token_set_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[6] = fuzz.token_sort_ratio(q1, q2)

    return fuzzy_features





def sent2vec(s):
    model = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\91701\\DataScience\\QuoraDuplicateQuestionpairs\\GoogleNews-vectors-negative300.bin", binary=True)
    stop_words = stopwords.words('english')
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v/np.sqrt((v ** 2).sum())


def process_word2vec_features_test(q1, q2):
    q1_vector = sent2vec(q1)
    q1_vector = np.nan_to_num(q1_vector)
    q2_vector = sent2vec(q2)
    q2_vector = np.nan_to_num(q2_vector)
    word2vec = [0.0] * 7
    if len(q1) == 0 or len(q2) == 0:
        return word2vec
    # cosine_distance
    word2vec[0] = cosine(q1_vector, q2_vector)
    # cityblock_distance
    word2vec[1] = cityblock(q1_vector, q2_vector)
    # jaccard_distance
    word2vec[2] = jaccard(q1_vector, q2_vector)
    # canberra_distance
    word2vec[3] = canberra(q1_vector, q2_vector)
    # euclidean_distance
    word2vec[4] = euclidean(q1_vector, q2_vector)
    # minkowski_distance
    word2vec[5] = minkowski(q1_vector, q2_vector)
    # braycurtis_distance
    word2vec[6] = braycurtis(q1_vector, q2_vector)

    return word2vec


def process_tag_features_test(q1, q2):
    q1_tokens = word_tokenize(q1)  # tokenization
    q2_tokens = word_tokenize(q2)
    features_teg = [0.0] * 3
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return features_teg

    # taking input text as India
    p1 = nltk.pos_tag(q1_tokens)

    # ans returns a list of tuple
    noun1 = []
    verb1 = []
    adj1 = []
    for t in p1:
        if t[1] == 'NN' or t[1] == 'NNS' or t[1] == 'NNPS' or t[1] == 'NNP':
            noun1.append(t[0])
        if t[1] == 'VB' or t[1] == 'VBG' or t[1] == 'VBN' or t[1] == 'VBD' or t[1] == 'VBP' or t[1] == 'VBZ':
            verb1.append(t[0])
        if t[1] == 'JJ' or t[1] == 'JJR' or t[1] == 'JJS':
            adj1.append(t[0])

    # Now for question2
    p2 = nltk.pos_tag(q2_tokens)

    # returns a list of tuple
    noun2 = []
    verb2 = []
    adj2 = []
    for t in p2:
        if t[1] == 'NN' or t[1] == 'NNS' or t[1] == 'NNPS' or t[1] == 'NNP':
            noun2.append(t[0])
        if t[1] == 'VB' or t[1] == 'VBG' or t[1] == 'VBN' or t[1] == 'VBD' or t[1] == 'VBP' or t[1] == 'VBZ':
            verb2.append(t[0])
        if t[1] == 'JJ' or t[1] == 'JJR' or t[1] == 'JJS':
            adj2.append(t[0])

    common_noun_count = len(list([word for word in noun1 if word in noun2]))
    common_verb_count = len(list([word for word in verb1 if word in verb2]))
    common_adj_count = len(list([word for word in adj1 if word in adj2]))
    features_teg[0] = common_noun_count
    features_teg[1] = common_verb_count
    features_teg[2] = common_adj_count

    return features_teg


def query_point_creator(q1, q2):
    input_query = []

    # preprocessing
    q1 = preprocessing(q1)
    q2 = preprocessing(q2)

    # fetch features
    features = process_features_test(q1, q2)
    input_query.extend(features)

    # fetch fuzzy features
    fuzzy_features = process_fuzzy_features_test(q1, q2)
    input_query.extend(fuzzy_features)

    # fetch word 2 vec
    word2vec = process_word2vec_features_test(q1, q2)
    input_query.extend(word2vec)
    # fetch feature tag
    features_teg = process_tag_features_test(q1, q2)
    input_query.extend(features_teg)

    # bow feature for q1
    q1 = tfidfvectorizer.transform([q1]).toarray()

    # bow feature for q2
    q2 = tfidfvectorizer.transform([q2]).toarray()

    return np.hstack((q1, q2, np.array(input_query).reshape(1, 27)))



