import sys

import nltk
# nltk.download('brown')
# nltk.download('universal_tagset')
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words, i):
    n = len(words)
    prev_word = '<s>' if i - 1 < 0 else words[i - 1]
    next_word = '</s>' if i + 1 >= n else words[i + 1]
    prev_skip = '<s>' if i - 2 < 0 else words[i - 2]
    next_skip = '</s>' if i + 2 >= n else words[i + 2]

    features = [
        f'prevbigram-{prev_word}',
        f'nextbigram-{next_word}',
        f'prevskip-{prev_skip}',
        f'nextskip-{next_skip}',
        f'prevtrigram-{prev_word}-{prev_skip}',
        f'nexttrigram-{next_word}-{next_skip}',
        f'centertrigram-{prev_skip}-{next_word}'
    ]

    return features


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    features = []

    features.append(f'word-{word}')

    if word.isupper():
        features.append('allcaps')
    elif word[0].isupper():
        features.append('capital')

    word_shape = ''.join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in word])
    features.append(f'wordshape-{word_shape}')

    short_word_shape = ''.join(['' if i>=1 and c==word_shape[i-1] else c for i,c in enumerate(word_shape)])
    features.append(f'short-wordshape-{short_word_shape}')

    if any(c.isdigit() for c in word):
        features.append('number')

    if '-' in word:
        features.append('hyphen')
    

    for i in range(1,5):
        if i <= len(word):
            features.append(f'prefix{i}-{word[:i]}')

    for i in range(1,5):
        if i <= len(word):
            features.append(f'suffix{i}-{word[-i:]}')

    return features


# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    features = get_ngram_features(words, i) + get_word_features(words[i])
    features.append(f'tagbigram-{prevtag}')
    features = [f.lower() if not f.startswith('wordshape') or f.startswith('short-wordshape') else f  for f in features]
    return features


# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    feature_count = {}
    for sentence in corpus_features:
        for word in sentence:
            for feature in word:
                if not feature in feature_count:
                    feature_count[feature] = 0
                feature_count[feature] += 1

    rare_features = set([feature for feature, count in feature_count.items() if count < threshold])
    common_features = set([feature for feature in feature_count if not feature in rare_features])

    for i, sentence in enumerate(corpus_features):
        for j, word in enumerate(sentence):
            corpus_features[i][j] = [feature for feature in word if feature in common_features]

    return corpus_features, common_features


# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    feature_dict = {feature: index for index, feature in enumerate(common_features)}

    tags = set(tag for sentence_tags in corpus_tags for tag in sentence_tags)
    tag_dict = {tag: index for index, tag in enumerate(tags)}

    return feature_dict, tag_dict

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    result = []
    for sentence_tags in corpus_tags:
        for word_tag in sentence_tags:
            result.append(tag_dict[word_tag])
    
    return numpy.array(result)

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    sample_count = 0
    rows = []
    cols = []
    values = []
    for sentence in corpus_features:
        for word in sentence:
            for feature in word:
                if feature in feature_dict:
                    rows.append(sample_count)
                    cols.append(feature_dict[feature])
                    values.append(1)
            sample_count += 1
    X = csr_matrix((values, (numpy.array(rows), numpy.array(cols))), shape=(sample_count, len(feature_dict)))
    return X


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    corpus_features = []
    corpus_sents, corpus_tags = load_training_corpus(proportion)
    for sentence in corpus_sents:
        sentence_features = []
        for j, word in enumerate(sentence):
            if j == 0:
                prevtag = '<S>'
            else:
                prevtag = corpus_tags[j - 1]
            sentence_features.append(get_features(sentence, j, prevtag))
        corpus_features.append(sentence_features)

    corpus_features, common_features = remove_rare_features(corpus_features)

    feature_dict, tag_dict = get_feature_and_label_dictionaries(common_features, corpus_tags)
    X = build_X(corpus_features, feature_dict)
    Y = build_Y(corpus_tags, tag_dict)
    model = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    model.fit(X, Y)

    return model, feature_dict, tag_dict



# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    Y_pred = numpy.empty((len(test_sent[0])-1, len(reverse_tag_dict), len(reverse_tag_dict)))
    for i, word in enumerate(test_sent[0]):
        if i > 0 :
            features = []
            for index, tag in reverse_tag_dict.items():
                features.append(get_features(test_sent[0], i, tag))
            X = build_X([features], feature_dict)  
            log_probabilities = model.predict_log_proba(X)
            Y_pred[i - 1] = log_probabilities
    
    prevtag = '<S>'
    features = get_features(test_sent[0], 0, prevtag)
    X = build_X([[features]], feature_dict) # T × F
    Y_start = model.predict_log_proba(X) # T × T

    return Y_start, Y_pred


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    n = Y_pred.shape[0]+1
    T = Y_start.shape[1]
    V = numpy.empty((n, T))
    BP = numpy.empty((n, T))
    V[0] = Y_start
    for i in range(1, n):
        for j in range(T):
            max_prob = -numpy.inf
            max_backpointer = -1

            for t2 in range(T):
                prob = V[i - 1, t2] + Y_pred[i - 1, t2, j]
                if prob > max_prob:
                    max_prob = prob
                    max_backpointer = t2

            V[i, j] = max_prob
            BP[i, j] = max_backpointer

    tag_sequence = []
    max_prob_index = numpy.argmax(V[n - 1]) 
    for i in range(n - 1, 0, -1):
        tag_sequence.append(max_prob_index)
        max_prob_index = numpy.argmax(V[i])
        
    tag_sequence.reverse()
    return tag_sequence


# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    test_corpus = load_test_corpus(corpus_path)
    reverse_tag_dict = {v: k for k, v in tag_dict.items()}
    predicted_tags_corpus = []
    for test_sentence in test_corpus:
        Y_start, Y_pred = get_predictions([test_sentence], model, feature_dict, reverse_tag_dict)
        predicted_tag_indices = viterbi(Y_start, Y_pred)
        predicted_tags = [reverse_tag_dict[i] for i in predicted_tag_indices]
        predicted_tags_corpus.append(predicted_tags)

    return predicted_tags_corpus



def main(args):
    model, feature_dict, tag_dict = train(0.25)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
