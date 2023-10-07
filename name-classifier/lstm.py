import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import time as time


""" 
Bidirectional LSTM model
Hyperparameters: embedding_dim, units and dropouts  
"""
def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
  model = Sequential([
      Embedding(num_alphabets, embedding_dim, input_length=name_length),
      Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
      Dense(1, activation="sigmoid")
  ])
  model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])
  return model


""" Clean up names and gender, and turn into numerical representation """
def preprocess(names_genders, train=True):

  # Step 0: delete non-alphabetical characters from 'name' column
  names_genders['name'] = names_genders['name'].str.replace(r'[^a-zA-Z]', '', regex=True)

  # Step 1: Lowercase
  names_genders['name'] = names_genders['name'].str.lower()

  # Step 2: Split individual characters
  names_genders['name'] = [list(name) for name in names_genders['name']]

  # Step 3: Pad names with spaces to make all names same length
  name_length = 50
  names_genders['name'] = [
      (name + [' ']*name_length)[:name_length] 
      for name in names_genders['name']
  ]

  # Step 4: Encode Characters to Numbers
  names_genders['name'] = [
      [
          max(0.0, ord(char)-96.0) 
          for char in name
      ]
      for name in names_genders['name']
  ]
  if train:
      # Step 5: Encode Gender to Numbers
      # 0 or 1
      names_genders['gender'] = [
          0.0 if gender=='F' else 1.0 
          for gender in names_genders['gender']
      ]

  return names_genders


""" Train a biderectional LSTM on a set of pre-processed data"""
def train_lstm_model(names_genders, train_split):

  t_train_start = time.time()
  
  # Step 1: Instantiate the model
  trained_lstm = lstm_model(num_alphabets=27, name_length=50, embedding_dim=256)
  
  # Step 2: Split Training and Test Data
  names = np.asarray(names_genders['name'].values.tolist())
  genders = np.asarray(names_genders['gender'].values.tolist())
  names_train, names_test, genders_train, genders_test = train_test_split(names, genders, train_size=train_split, random_state=0)
  
  # Step 3: Train the model
  callbacks = [
      EarlyStopping(monitor='val_accuracy',
                    min_delta=1e-3,
                    patience=5,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1),
  ]
  history = trained_lstm.fit(x=names_train,
                            y=genders_train,
                            batch_size=64,
                            epochs=50,
                            validation_data=(names_test, genders_test),
                            callbacks=callbacks)
  print('Time to train LSTM:', round(time.time() - t_train_start, 3), '[s]')
  
  # Step 4: Plot training history
  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

  return trained_lstm


""" LSTM classifier """
class LSTMNameClassifier:
  
  """ Initalize a LSTM model (load from file or train a new one) """
  def __init__(self, model_to_load=None, train_data='name_gender.csv', train_split=.8):
    if model_to_load:
      # load previously trained model from file
      try:
        self.trained_lstm = load_model('saved_models/' + model_to_load + '.h5')
      except:
        raise Exception('Try LSTMNameClassifier(model_to_load=\'lstm_name_gender_split80\')')
    else:
      # train a new model from scratch
      # load and preprocess data 
      names_genders = pd.read_csv(train_data, dtype=str)
      names_genders = names_genders.fillna(0)
      names_genders = preprocess(names_genders)
      # train (and save) the model
      self.trained_lstm = train_lstm_model(names_genders, train_split)
      # Save the model
      self.trained_lstm.save('saved_models/lstm_' + train_data.split('.')[0] + '.h5')
  

  """ Test trained model using a list of names, or test_data from a file """
  def test_lstm_model(self, names=[], test_data='name_gender.csv'):
    
    if names:
      # Convert to dataframe
      names_genders = pd.DataFrame({'name': names})
    
    else:
      # load data
      names_genders = pd.read_csv(test_data, dtype=str)
      names_genders = names_genders.fillna(0)
      names_genders = preprocess(names_genders)
    
    # Preprocess
    names_genders = preprocess(names_genders, train=False)
    # Predictions
    result = self.trained_lstm.predict(np.asarray(
        names_genders['name'].values.tolist())).squeeze(axis=1)
    
    names_genders['M/F'] = [
        'Male' if logit > 0.5 else 'Female' for logit in result
    ]
    names_genders['Probability'] = [
        logit if logit > 0.5 else 1.0 - logit for logit in result
    ]

    # Format the output
    names_genders['name'] = names
    names_genders.rename(columns={'name': 'Name'}, inplace=True)
    names_genders['Probability'] = names_genders['Probability'].round(3)
    names_genders.drop_duplicates(inplace=True)

    print(names_genders.head())
    return names_genders

      

