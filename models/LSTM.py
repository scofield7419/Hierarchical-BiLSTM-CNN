# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import Input, merge, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.models import Model
from keras.optimizers import *
from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding, SimpleRNN, LSTM, GRU
import os
import h5py
from models.LSTM_CNN_data_preprocessing import get_data
import shutil

######### parameters:
np.random.seed(1337)  # for reproducibility
batch_size = 16  # 8
epochs = 3  # 100

# texts
max_features = 112000  # 词典长度
maxlen = 1024 * 5  # 512  # 一条句子的token最长长度（max sequence length）
embedding_dims = 100  # 200

# lstm
time_step = maxlen
lstm_hidden_dim = 1024

# mlp
hidden_dim_1 = 20
output_dim = 3  # 三分类，做出二进制标签，只需2维

##########
test_len = 128
validation_fold = 0.1

########## 准备数据
data_path = r'./data/'
save_path = r'./res-CNN-LSTM/'
save_data_path_selected_data_path = save_path
if not os.path.exists(save_data_path_selected_data_path):
    os.mkdir(save_data_path_selected_data_path)
file_str = "handled_reviews.csv"
file_dir = os.path.join(data_path, file_str)

# embedding_file = r'embeddings-200d.txt'
embedding_file = r'embeddings-100d.txt'
embedding_file = os.path.join(data_path, embedding_file)

word_vocab, return_str, (x_train_text, y_train) = get_data(file_dir,
                                                           embedding_file,
                                                           save_data_path_selected_data_path,
                                                           maxlen=maxlen,
                                                           )
max_features = len(word_vocab)
# print(max_features)

#########prepare embeddings
embeddings_index = {}
f = open(embedding_file)
ii = 0
for line in f:
    if ii == 0:
        ii += 1
        continue
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[ii - 1] = coefs
    ii += 1
f.close()

##word_index: 词表
embedding_matrix = np.zeros((max_features + 1, embedding_dims))
i = 0
for word in word_vocab:
    embedding_vector = embeddings_index.get(i)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    i += 1

###喂入词token对应的id即可
text_input = Input(shape=(maxlen,), dtype='int32', name="input")

embeddeds = Embedding(max_features + 1, embedding_dims, input_length=maxlen,
                      weights=[embedding_matrix], trainable=False)(text_input)
biLSTM = Bidirectional(LSTM(lstm_hidden_dim, dropout=0.5))(embeddeds)
hiden1 = Dense(hidden_dim_1)(biLSTM)
hiden1 = Dropout(0.5)(hiden1)
hiden1 = Activation('relu')(hiden1)
hiden2 = Dense(output_dim)(hiden1)
output = Activation('softmax')(hiden2)

######训练
model = Model(input=text_input, output=output)

print(model.summary())


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-8)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy', precision, recall, f1])

hist = model.fit(x_train_text, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 shuffle=True,
                 verbose=1,
                 validation_split=validation_fold)

#########logging
val_acc = hist.history['val_acc']
str_0 = "Train on %d samples, validate on %d samples" % (len(y_train) * (1 - validation_fold),
                                                         len(y_train) * validation_fold)
str_0_1 = "epochs: %d" % epochs
str_2 = "最后的val score ：%4f" % val_acc[len(val_acc) - 1]
print(str_2)
store_str = "\n--------------------\n" + "///" + file_str + "\n" + return_str + str_0 + "\n\n" + str_0_1 + "\n\n" + str(
    hist.history) + "\n" + str_2 + "\n\n" + "\n"  # + str_4 + "\n"

with open(save_data_path_selected_data_path + "/log_lstm.txt", 'a+') as f:
    f.write(store_str)

    # model.save('my_model.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
    # model = keras.models.load_model('my_model.h5')
