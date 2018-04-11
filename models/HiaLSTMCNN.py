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
from models.HLC_data_preprocessing import get_data
import shutil

######### parameters:
np.random.seed(1337)  # for reproducibility
batch_size = 512  # 8
epochs = 3  # 100

# texts
max_features = 112000  # 词典长度
maxlen = 256  # 512  # 一条句子的token最长长度（max sequence length）
embedding_dims = 100  # 200

# cnn
cnn_module_num = 20
filters = 5
kernel_size = 3  # 3-gram
output_repre_dim = 100  # 200

# lstm
time_step = cnn_module_num
lstm_hidden_dim = 256

# bgmlp
input_fea_dim = 20
bg_hidden_dim = 20  # 50

# mlp
hidden_dim_1 = 20
output_dim = 3  # 三分类，若做出二进制标签，只需2维，或者（0，0，1）

##########
test_len = 128
validation_fold = 0.1

########## 准备数据
data_path = r'./data/'
save_path = r'./res-HLC/'
# sub_file_name = "8-2-fold"
save_data_path_selected_data_path = save_path
if not os.path.exists(save_data_path_selected_data_path):
    os.mkdir(save_data_path_selected_data_path)
file_str = "handled_reviews.csv"
file_dir = os.path.join(data_path, file_str)

# embedding_file = r'embeddings-200d.txt'
embedding_file = r'embeddings-100d.txt'
embedding_file = os.path.join(data_path, embedding_file)

word_vocab, return_str, (x_train_text, x_train_feat, y_train) = get_data(file_dir,
                                                                         embedding_file,
                                                                         save_data_path_selected_data_path,
                                                                         maxlen=maxlen,
                                                                         gb_len=input_fea_dim,
                                                                         test_len=test_len)
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
    # embeddings_index[word] = coefs
    embeddings_index[ii - 1] = coefs
    ii += 1
f.close()

# print('Found %s word vectors.' % len(embeddings_index))

##word_index: 词表
##embedding_matrix: 变量
##embedding_matrix的长度多一行，是的不存在embedding的词的值都为0
embedding_matrix = np.zeros((max_features + 1, embedding_dims))
i = 0
for word in word_vocab:
    # embedding_vector = embeddings_index.get(word)
    embedding_vector = embeddings_index.get(i)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    i += 1


######### CNN
def get_CNN(index):
    ###喂入词token对应的id即可
    text_input = Input(shape=(maxlen,), dtype='int32', name=("cnn" + str(index)))

    embeddeds = Embedding(max_features + 1, embedding_dims, input_length=maxlen,
                          weights=[embedding_matrix], trainable=False)(text_input)
    convd = Conv1D(filters,
                   kernel_size,
                   padding='valid',
                   activation='relu',
                   strides=1)(embeddeds)
    dropped = Dropout(0.5)(convd)
    maxp = GlobalMaxPooling1D()(dropped)
    densed = Dense(output_repre_dim, activation='relu')(maxp)
    return text_input, densed


##########拼接CNN
cnnzs = []
text_inputs = []
for cnn_layer in range(cnn_module_num):
    cnn_text_input, a_cnn_result = get_CNN(cnn_layer)
    cnnzs.append(a_cnn_result)
    text_inputs.append(cnn_text_input)
merged1 = merge(cnnzs, mode='concat')  # , concat_axis=1)
reshaped = Reshape((cnn_module_num, output_repre_dim), input_shape=(output_repre_dim * cnn_module_num,))(merged1)

#########LSTM
biLSTM = Bidirectional(LSTM(lstm_hidden_dim, dropout=0.5))(reshaped)
lstm_densed = Dense(32, activation='relu')(biLSTM)

#########融合MLP bg信息
input_feat = Input(shape=(input_fea_dim,), dtype='float32', name="mlp")
hiden_feat = Dense(bg_hidden_dim)(input_feat)
combined = merge([lstm_densed, hiden_feat], mode='concat')

#########MLP
hiden1 = Dense(hidden_dim_1)(combined)
hiden1 = Dropout(0.5)(hiden1)
hiden1 = Activation('relu')(hiden1)
hiden2 = Dense(output_dim)(hiden1)
output = Activation('softmax')(hiden2)

######训练
# text_inputs 外包了一层list, 即maxlen 外还有个cnn_module_num20的list
text_inputs.append(input_feat)
input_all = text_inputs
model = Model(input=input_all, output=output)

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

x_train_input_dict = {}
for intem in range(len(x_train_text)):
    key_name = "cnn" + str(intem)
    x_train_input_dict[key_name] = x_train_text[intem]
x_train_input_dict["mlp"] = x_train_feat

hist = model.fit(x_train_input_dict, y_train,
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

with open(save_data_path_selected_data_path + "/log.txt", 'a+') as f:
    f.write(store_str)

