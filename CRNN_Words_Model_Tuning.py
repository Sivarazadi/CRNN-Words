# imports
from distutils.command.build import build
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, LSTM
from string import ascii_letters
from string import digits
import fnmatch
import cv2
import numpy as np
from keras.utils import pad_sequences
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time
import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner import HyperParameters
import pickle
from alive_progress import alive_bar
from time import sleep
from tqdm import tqdm
import dill




# list of lower case, upper case, and numbered characters
char_list = ascii_letters+digits





# encoding each character in a text into digits for NN to read characters as digits
def encode(txt):

    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst




def load_img(num_total_img, training_ratio = 0.9, dev_ratio=0.05, test_ratio=0.05):
    ''' function that loads in images and creates a training/dev/test split with img, original txt, encoded txt, input length, and label length  '''
    
    path = 'Desktop\\mnt\\ramdisk\\max\\90kDICT32px'


    # lists for training dataset
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    orig_txt = []
    
    #lists for devation dataset
    dev_img = []
    dev_txt = []
    dev_input_length = []
    dev_label_length = []
    dev_orig_txt = []

    #lists for test dataset
    test_img = []
    test_txt = []
    test_input_length = []
    test_label_length = []
    test_orig_txt = []



    # max label length for equal sized encodings
    global max_label_len 
    max_label_len = 0

    training_size = int(num_total_img * training_ratio)
    dev_size = round(num_total_img * dev_ratio)
    test_size = round(num_total_img * test_ratio)

    i = 1 
    flag = 0

    with tqdm(total=num_total_img) as pbar: # progress bar
        for root, dirnames, filenames in os.walk(path):
            for f_name in fnmatch.filter(filenames, '*.jpg'):
                img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)   
                #print(f_name)
                # convert each image of shape (32, 128, 1)
                w, h = img.shape
                if h > 128 or w > 32:
                    continue
                if w < 32:
                    add_zeros = np.ones((32-w, h))*255
                    img = np.concatenate((img, add_zeros))
        
                if h < 128:
                    add_zeros = np.ones((32, 128-h))*255
                    img = np.concatenate((img, add_zeros), axis=1)
                img = np.expand_dims(img , axis = 2)
                
                # Normalize each image
                img = img/255.
                
                # get the text from the image
                txt = f_name.split('_')[1]

                # compute maximum length of the text
                if len(txt) > max_label_len:
                    max_label_len = len(txt)
                    
                
                # split data into devation and training dataset
                if i < dev_size+1:     
                    dev_orig_txt.append(txt)   
                    dev_label_length.append(len(txt))
                    dev_input_length.append(31)
                    dev_img.append(img)
                    dev_txt.append(encode(txt))
                if i in range(dev_size, test_size):
                    test_txt.append(txt)   
                    test_label_length.append(len(txt))
                    test_input_length.append(31)
                    test_img.append(img)
                    test_txt.append(encode(txt)) 
                if i in range(dev_size+test_size, num_total_img):
                    orig_txt.append(txt)   
                    train_label_length.append(len(txt))
                    train_input_length.append(31)
                    training_img.append(img)
                    training_txt.append(encode(txt)) 

                if i == num_total_img:
                    flag = 1
                    break
                i+=1
            
                pbar.update(1)

            if flag == 1:
                break
        
    return training_img, training_txt, train_input_length, train_label_length, orig_txt, dev_img, dev_txt, dev_input_length, dev_label_length, dev_orig_txt, test_img, test_txt, test_input_length, test_label_length, test_orig_txt









def pad_txt(txt):
    padded_txt = pad_sequences(txt, maxlen=max_label_len, padding='post', value = len(char_list))
    return padded_txt



def build_model(hp):
# input with shape of height=32 and width=128 
    global inputs, outputs

    inputs = Input(shape=(32,128,1))
    
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(hp.Int('input_units (conv_1)', min_value=64, max_value=256, step=64), (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    
    conv_2 = Conv2D(hp.Int('conv_2', min_value=128, max_value=448, step=64), (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    
    conv_3 = Conv2D(hp.Int('conv_3', min_value=256, max_value=512, step=64), (3,3), activation = 'relu', padding='same')(pool_2)
    
    conv_4 = Conv2D(hp.Int('conv_4', min_value=64, max_value=384, step=64), (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    
    conv_5 = Conv2D(hp.Int('conv_5', min_value=64, max_value=512, step=64), (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)
    
    conv_6 = Conv2D(hp.Int('conv_6', min_value=64, max_value=320, step=64), (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    
    conv_7 = Conv2D(hp.Int('conv_7', min_value=128, max_value=512, step=64), (2,2), activation = 'relu')(pool_6)
    
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(hp.Int('blstm_1', min_value=128, max_value=512, step=64), return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(hp.Int('blstm_2', min_value=320, max_value=512, step=64), return_sequences=True, dropout = 0.2))(blstm_1)
    
    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)



    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
    
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    #model to be used at training time
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam', metrics=['accuracy'])

    return model


LOG_DIR = f"{int(time.time())}"



training_img, training_txt, train_input_length, train_label_length, orig_txt, dev_img, dev_txt, dev_input_length, dev_label_length, dev_orig_txt, test_img, test_txt, test_input_length, test_label_length, test_orig_txt = load_img(num_total_img=50000)

train_padded_txt = pad_txt(training_txt)
dev_padded_txt = pad_txt(dev_txt)
test_padded_txt = pad_txt(test_txt)

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

dev_img = np.array(dev_img)
dev_input_length = np.array(dev_input_length)
dev_label_length = np.array(dev_label_length)

test_img = np.array(test_img)
test_input_length = np.array(test_input_length)
test_label_length = np.array(test_label_length)

print(orig_txt[1])

print(training_txt[1])

print(len(training_txt))
print(len(dev_txt))

print(train_padded_txt[0])


batch_size = 256
epochs = 7
len_train = len(training_img)
len_dev = len(dev_img)
len_test = len(test_img)


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=40,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR)


# running tuner to find best hyperparameters
tuner.search(x=[training_img, train_padded_txt, train_input_length, train_label_length], 
             y=np.zeros(len_train),
             epochs = epochs,
             batch_size= batch_size,
             validation_data = ([dev_img, dev_padded_txt, dev_input_length, dev_label_length], [np.zeros(len_dev)])
             )

# dumping into pickle
with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    dill.dump(tuner, f)