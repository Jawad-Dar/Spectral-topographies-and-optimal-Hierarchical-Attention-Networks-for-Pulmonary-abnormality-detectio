import random
import string
import numpy as np
import pandas as pd
from nltk import tokenize
from keras import backend as K
from nltk.corpus import stopwords
from keras.engine.topology import Layer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras import initializers as initializers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Sequential, Model, load_model
import os, logging, warnings, csv
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
import WCSO_HAN.WCSO

def classify(X,Y,dst,tr, acc, TPR, TNR):
    tpr = tr/100
    def read_data(filename):
        data = []
        with open(filename, newline='') as f:
            reader = csv.reader(f)  # full data
            for row in reader:  # row
                tem = ""
                for col in row:  # data in row
                    tem += ((col))  # each row data
                data.append(tem)
        return data

    def read_label(filename):  # read the 1D data
        cl = []
        with open(filename, newline='') as f:
            reader = csv.reader(f)  # full data
            for row in reader:  # row
                for col in row:  # data in row
                    cl.append(float(col))  # each row data
        return cl

    label = "Main//Preprocessed//"+dst + "_label.csv"
    feat = "Main//Preprocessed//"+dst + 'P1_Feature.csv'
    data = read_data(feat)
    item = read_label(label)
    np.savetxt("glove_edited.txt", data, delimiter=' ', fmt='%s')

    articles = []
    for i in range(len(data)): articles.append((data[i], item[i]))
    data_df = pd.DataFrame(data=articles, columns=['Text', "Category"])

    def cleanString(review, stopWords):
        lemmatizer = WordNetLemmatizer()
        returnString = ""
        sentence_token = tokenize.sent_tokenize(review)
        idx_list = []
        for j in range(len(sentence_token)):
            single_sentence = tokenize.word_tokenize(sentence_token[j])
            sentences_filtered = [(idx, lemmatizer.lemmatize(w.lower())) for idx, w in enumerate(single_sentence)
                                  if w.lower() not in stopWords and w.isalnum()]
            idx_list.append([x[0] for x in sentences_filtered])
            word_list = [x[1] for x in sentences_filtered]
            returnString = returnString + ' '.join(word_list) + ' . '

        return returnString, idx_list

    def split_df(dataframe, column_name, training_split=0.6, validation_split=0.2, test_split=0.2):
        """
        Splits a pandas dataframe into trainingset, validationset and testset in specified ratio.
        All sets are balanced, which means they have the same ratio for each categorie as the full set.
        Input:   dataframe        - Pandas Dataframe, should include a column for data and one for categories
                 column_name      - Name of dataframe column which contains the categorical output values
                 training_split   - from ]0,1[, default = 0.6
                 validation_split - from ]0,1[, default = 0.2
                 test_split       - from ]0,1[, default = 0.2
                                    Sum of all splits need to be 1
        Output:  train            - Pandas DataFrame of trainset
                 validation       - Pandas DataFrame of validationset
                 test             - Pandas DataFrame of testset
        """
        if training_split + validation_split + test_split != 1.0:
            raise ValueError('Split paramter sum should be 1.0')

        total = len(dataframe.index)

        train = dataframe.reset_index().groupby(column_name).apply(lambda x: x.sample(frac=training_split)) \
            .reset_index(drop=True).set_index('index')
        train = train.sample(frac=1)
        temp_df = dataframe.drop(train.index)
        validation = temp_df.reset_index().groupby(column_name) \
            .apply(lambda x: x.sample(frac=validation_split / (test_split + validation_split))) \
            .reset_index(drop=True).set_index('index')
        validation = validation.sample(frac=1)
        test = temp_df.drop(validation.index)
        test = test.sample(frac=1)
        return train, validation, test

    def wordToSeq(text, word_index, max_sentences, max_words, max_features):
        """
        Converts a string to a numpy matrix where each word is tokenized.
        Arrays are zero-padded to max_sentences and max_words length.

        Input:    text           - string of sentences
                  word_index     - trained word_index
                  max_sentences  - maximum number of sentences allowed per document for HAN
                  max_words      - maximum number of words in each sentence for HAN
                  max_features   - maximum number of unique words to be tokenized
        Output:   data           - Numpy Matrix of size [max_sentences x max_words]
        """
        sentences = tokenize.sent_tokenize(text)
        data = np.zeros((max_sentences, max_words), dtype='int32')
        for j, sent in enumerate(sentences):
            if j < max_sentences:
                wordTokens = tokenize.word_tokenize(sent.rstrip('.'))
                wordTokens = [w for w in wordTokens]
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_words and word_index[word] < max_features:
                            data[j, k] = word_index[word]
                            k = k + 1
                    except:
                        pass
        return data

    def to_categorical(series, class_dict):
        """
        Converts category labels to vectors,
        Input:     series     - pandas Series containing numbered category labels
                   class_dict - dictionary of integer to category string
                                e.g. {0: 'business', 1: 'entertainment', 2: 'politics', 3: 'sport', 4: 'tech'}
        Output:    Array      - numpy array containing categories converted to lists
                                e.g. 0:'business'      -> [1 0 0 0 0]
                                     1:'entertainment' -> [0 1 0 0 0]
                                     2:'politics'      -> [0 0 1 0 0]
                                     3:'sport'          -> [0 0 0 1 0]
                                     4:'tech'          -> [0 0 0 0 1]
        """
        n_classes = len(class_dict)
        new_dict = {}
        for key, value in class_dict.items():
            cat_list = [0] * n_classes
            cat_list[key] = 1
            new_dict[key] = cat_list
        y_cat = []
        for key, value in series.iteritems():
            y_cat.append(new_dict[value])
        return np.array(y_cat)

    # Attention layer
    class AttentionLayer(Layer):
        """
        Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
        - Yang et. al.
        Theano backend
        """

        def __init__(self, attention_dim=100, return_coefficients=False, **kwargs):
            # Initializer
            self.supports_masking = True
            self.return_coefficients = return_coefficients
            self.init = initializers.get('glorot_uniform')  # initializes values with uniform distribution
            self.attention_dim = attention_dim
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            # Builds all weights
            # W = Weight matrix, b = bias vector, u = context vector
            assert len(input_shape) == 3
            self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
            self.b = K.variable(self.init((self.attention_dim,)), name='b')
            self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
            self.trainable_weights = [self.W, self.b, self.u]

            super(AttentionLayer, self).build(input_shape)

        def compute_mask(self, input, input_mask=None):
            return None

        def call(self, hit, mask=None):
            # Here, the actual calculation is done
            uit = K.bias_add(K.dot(hit, self.W), self.b)
            uit = K.tanh(uit)

            ait = K.dot(uit, self.u)
            ait = K.squeeze(ait, -1)
            ait = K.exp(ait)

            if mask is not None:
                ait *= K.cast(mask, K.floatx())

            ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            ait = K.expand_dims(ait)
            weighted_input = hit * ait

            if self.return_coefficients:
                return [K.sum(weighted_input, axis=1), ait]
            else:
                return K.sum(weighted_input, axis=1)

        def compute_output_shape(self, input_shape):
            if self.return_coefficients:
                return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
            else:
                return input_shape[0], input_shape[-1]

    """
    Compute average number of words in each sentence and average number of sentences in each document.
    """
    n_sent = 0
    n_words = 0
    for i in range(data_df.shape[0]):
        sent = tokenize.sent_tokenize(data_df.loc[i, 'Text'])

        for satz in sent:
            n_words += len(tokenize.word_tokenize(satz))
        n_sent += len(sent)

    # Parameters
    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 10  # maximum number of sentences in one document
    MAX_WORD_NUM = 1  # maximum number of words in each sentence
    EMBED_SIZE = 5  # vector size of word embedding
    x_train, x_test, Y_train, y_Test = train_test_split(X, Y, train_size=tpr)
    # Data Preprocessing
    articles = []
    n = data_df['Text'].shape[0]
    col_number = data_df.columns.get_loc('Text')
    stopWords = set(stopwords.words('english'))
    data_cleaned = data_df.copy()
    for i in range(n):
        temp_string, idx_string = cleanString(data_df.iloc[i, col_number], stopWords)
        articles.append(temp_string)

    data_cleaned.loc[:, 'Text'] = pd.Series(articles, index=data_df.index)
    data_cleaned.loc[:, 'Category'] = pd.Categorical(data_cleaned.Category)
    data_cleaned['Code'] = data_cleaned.Category.cat.codes
    categoryToCode = dict(enumerate(data_cleaned['Category'].cat.categories))

    # Tokenization

    texts = []
    n = data_cleaned['Text'].shape[0]
    for i in range(n):
        s = data_cleaned['Text'].iloc[i]
        s = ' '.join([word.strip(string.punctuation) for word in s.split() if word.strip(string.punctuation) is not ""])
        texts.append(s)
    tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True, oov_token=None)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    # GloVe Embedding Matrix

    # Load word vectors from pre-trained dataset
    embeddings_index = {}
    f = open(os.path.join(os.getcwd(), 'glove_edited.txt'), encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    def prediction(x_test,y):
        pred = model.predict(x_test)
        pred = pred.flatten()
        y_pred = []
        for i in range(len(pred)): y_pred.append(round(pred[i]))
        predict = np.concatenate((Y_train,y_pred))
        target = np.concatenate((Y_train,y_Test))
        return predict,target



    # Search words in our word index in the pre-trained dataset
    # Create an embedding matrix for our bbc dataset
    min_wordCount = 0
    absent_words = 0
    small_words = 0
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    word_counts = tokenizer.word_counts
    for word, i in word_index.items():
        if word_counts[word] > min_wordCount:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                absent_words += 1
        else:
            small_words += 1

    # Splitting Data Set
    train, validation, test = split_df(data_cleaned, 'Code', tpr,(1-tpr)/2, (1-tpr)/2)

    # Training
    paras = []
    for i in range(train['Text'].shape[0]):
        sequence = wordToSeq(train['Text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_train = np.array(paras)
    y_train = to_categorical(train['Code'], categoryToCode)

    # Validation
    paras = []
    for i in range(validation['Text'].shape[0]):
        sequence = wordToSeq(validation['Text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_val = np.array(paras)
    y_val = to_categorical(validation['Code'], categoryToCode)

    # Test
    paras = []
    for i in range(test['Text'].shape[0]):
        sequence = wordToSeq(test['Text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_test = np.array(paras)
    y_test = to_categorical(test['Code'], categoryToCode)

    # HAN MODEL
    """
    Create Keras functional model for hierarchical attention network
    """
    embedding_layer = Embedding(len(word_index) + 1, EMBED_SIZE, weights=[embedding_matrix],
                                input_length=MAX_WORD_NUM, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(MAX_WORD_NUM,), dtype='int32', name='word_input')
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(GRU(50, return_sequences=True), name='word_gru')(word_sequences)
    word_dense = Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(EMBED_SIZE, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(MAX_SENTENCE_NUM, MAX_WORD_NUM), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(GRU(50, return_sequences=True), name='sent_gru')(sent_encoder)
    sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(EMBED_SIZE, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = Dropout(0.5, name='sent_dropout')(sent_att)
    preds = Dense(len(y_train[0]), activation='softmax', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    weight = np.array(model.get_weights())
    model.set_weights(weight * WCSO_HAN.WCSO.algm())

    # Train
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=50, verbose=0)
    y_pred,target = prediction(x_test,y_test)

    pred_val = np.unique(y_pred)
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)  # unique label
    for i1 in range(len(uni)):
        c = uni[i1]
        for i in range(len(target)):
            if (target[i] == c and y_pred[i] == c):
                tp = tp + 1
            if (target[i] != c and y_pred[i] != c):
                tn = tn + 1
            if (target[i] == c and y_pred[i] != c):
                fn = fn + 1
            if (target[i] != c and y_pred[i] == c):
                fp = fp + 1

    tn = tn / len(pred_val)
    fn = fn / pred_val[len(pred_val) - 1]
    fp = fp / pred_val[len(pred_val) - 1]
    tn = tn / len(uni)
    TPR.append(tp / (tp + fn))
    TNR.append(tn / (tn + fp))
    acc.append((tp + tn) / (tp + tn + fp + fn))
