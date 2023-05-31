# %%
import pandas as pd
import numpy as np
import re
import nltk

#nltk.download('stopwords')

# %%
df_cab = pd.read_csv('rideshare_kaggle.csv')
df_cab.head()

# %%
summary = df_cab['long_summary'].iloc[100:110]

# %%
summary.reset_index(drop=True)

# %%
summary = np.array(summary)

# %%
labels = ['cloudy', 'rainy', 'foggy', 'rainy', 'cloudy', 'foggy', 'rainy', 'foggy', 'cloudy', 'cloudy']
df_weather = pd.DataFrame({'Description': summary,
                           'Category': labels})
df_weather = df_weather[['Description', 'Category']]

# %%
df_weather

# %%
# TOKENIZER
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special chars and whitespace
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

# %%
norm_weather = normalize_corpus(summary)

# %%
norm_weather

# %%
# BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_weather)
cv_matrix = cv_matrix.toarray()

# %%
cv_matrix

# %%
vocab = cv.get_feature_names_out()
pd.DataFrame(cv_matrix, columns=vocab)

# %%
# BAG OF N-GRAMS MODEL
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_weather)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names_out()
pd.DataFrame(bv_matrix, columns=vocab)

# %%
# TF-IDF MODEL
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_weather)
tv_matrix = tv_matrix.toarray()

# %%
vocab = tv.get_feature_names_out()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

# %%
# DOCUMENT SIMILARITY - COSINE SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df

# %%
# CLUSTERING DOCUMENTS USING SIMILARITY FEATURES - KMEANS (EUCLIDEAN DISTANCE)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)
km.fit_transform(similarity_df)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['Cluster_Label'])
pd.concat([df_weather, cluster_labels], axis=1)

# %%
# TOPIC MODELS
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3, max_iter=100, random_state=0)
dt_matrix = lda.fit_transform(tv_matrix)
features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
features

# %%
# from the output, we discovered that
# T1 = foggy, T2 = rainy, T3 = cloudy

# %%
tt_matrix = lda.components_
for topic_weights in tt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 0.6]
    print(topic)
    print()

# %%
# CLUSTERING DOCUMENTS USING TOPIC MODEL FEATURES
km = KMeans(n_clusters=3)
km.fit_transform(features)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['Cluster_Labels'])
pd.concat([df_weather, cluster_labels], axis=1)

# %%
# WORD EMBEDDINGS
from gensim.models import word2vec

# %%
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in norm_weather]

# %%
print(tokenized_corpus)

# %%
# Set values for various parameters
vector_size = 10 # word vector dimensionality
window_context = 10 # context window size
min_word_count = 1 # min word count
sample = 1e-3 # downsample setting for frequent words

w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size=vector_size,
                              window=window_context, min_count=min_word_count,
                              sample=sample)

# %%
w2v_model.wv.index_to_key

# %%
tokenized_corpus

# %%
w2v_model.wv['foggy']

# %%
def avg_word_vector(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features), dtype='float64')
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

def avg_word_vectorizer(corpus, model, num_feature):
    vocabulary = set(model.wv.index_to_key)
    features = [avg_word_vector(tokenized_sentence, model, vocabulary, num_feature)
                for tokenized_sentence in corpus]
    
    return np.array(features)

# %%
w2v_feature_array = avg_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,
                                        num_feature=vector_size)
pd.DataFrame(w2v_feature_array)

# %%
from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation()
ap.fit(w2v_feature_array)
cluster_labels = ap.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['cluster_labels'])

# %%
df_new_weather = pd.concat([df_weather, cluster_labels], axis=1)

# %%
df_new_weather


