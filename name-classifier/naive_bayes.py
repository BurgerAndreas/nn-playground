import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import time as time


class NaiveBayesNameClassifier:
  """ Initialize the model """
  def __init__(self, model_to_load=None, train_data='name_gender.csv', train_split=.8):
    # If a saved model is specified, load model from file
    if model_to_load:
      try:
        with open('saved_models/' + str(model_to_load) + '.pickle', 'rb') as handle:
          self.clf_model, self.fitted_vectorizer = pickle.load(handle)

      except FileNotFoundError:
        print('There is no such saved model as' + model_to_load + 
          '.\nTry NameClassifier(load_model=nbayes_name_gender_split80)')
      self.data = None

    # Train a new model on train_data 
    else:
      # Load the data
      self.data, self.fitted_vectorizer = self._preprocess_data(train_data, train_split)
      self.clf_model = self.train_nbayes_model()

      # Save the model
      model_name = 'nbayes_' + train_data.split('.')[0] + '_split' + str(int(train_split*100))
      with open('saved_models/' + model_name + '.pickle', 'wb') as handle:
        pickle.dump([self.clf_model, self.fitted_vectorizer], handle, protocol=pickle.HIGHEST_PROTOCOL)


  """ Remove unwanted characters and represent it as Bag-Of-Words """
  def _preprocess_data(self, train_data, train_split):
    # Load the data 
    names_genders = pd.read_csv(train_data, dtype=str)
    names_genders = names_genders.fillna(0)

    # Delete non-alphabetical characters from 'name' column
    # (CountVectorizer lowercases later automatically)
    names_genders['name'] = names_genders['name'].str.replace(r'[^a-zA-Z]', '', regex=True)

    # convert text to numerical data using Bag-of-Words bigram
    char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    names_numerical = char_vectorizer.fit_transform(names_genders['name'])
    names_numerical = names_numerical.tocsc()  # more efficient for column-vectors
    genders_numerical = (names_genders.gender == 'M').values.astype(np.int) 

    # return (names_train, names_test, genders_train, genders_test), fitted_vectorizer
    return train_test_split(names_numerical, genders_numerical, train_size=train_split), char_vectorizer


  """ Test a single name on a trained / loaded model """
  def test_nbayes_name(self, name='Bernhard'):
    # Transform name into numerical format
    name_numerical = self.fitted_vectorizer.transform([str(name)])

    # Decide on male or female
    gender_predicted = self.clf_model.predict(name_numerical)
    if (gender_predicted == 1):
        print('NaiveBayes says', name, "is most likely a male name!")
    else:
        print('NaiveBayes says', name, "is most likely a female name!")

    return gender_predicted


  """ Train a model on pre-processed data """
  def train_nbayes_model(self):
    # train the model
    t_train_start = time.time()
    clf_model = MultinomialNB(alpha = 1)
    clf_model.fit(self.data[0], self.data[2])

    print('Time to train NaiveBayes:', round(time.time() - t_train_start, 3), '[s]')
    print('Training accuracy:', round(clf_model.score(self.data[0], self.data[2]), 3))

    return clf_model


  """ Test trained / loaded model on a set of data """
  def test_nbayes_data(self, test_data=None, train_split=.8):
    if test_data:
      # Use specified test_data
      data, _ = self._preprocess_data(test_data, train_split)

      # Return test accuracy
      return round(self.clf_model.score(data[1], data[3]), 3)

    else:
      # Try to use same data as for training
      if self.data is None:
        # There was no training, model was loaded from file
        raise Exception('NameClassifier has no training data to work with. \
          If NameClassifier was initialized using load_model, test_data in NameClassifier.test_model() must be specified. \
          Try NameClassifier.test_model(\'name_gender.csv\').')

      # return test accuracy
      return round(self.clf_model.score(self.data[1], self.data[3]), 3)
