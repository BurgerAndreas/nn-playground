from naive_bayes import NaiveBayesNameClassifier
from lstm import LSTMNameClassifier


""" Test the already trained NaiveBayes model on your custom data """
print('\nTesting the pre-trained NaiveBayes model')

# model was trained on 80% of name_gender.csv
loaded_nbayes = NaiveBayesNameClassifier(model_to_load='nbayes_name_gender_split80')

# try it on your custom data
#loaded_nbayes.test_model(train_data='my_custom_data.csv')

# try out a name
loaded_nbayes.test_nbayes_name(name='Bernhard')


""" Train and test NaiveBayes from scratch """
# model is trained on 80% of name_gender.csv
#trained_nbayes = NaiveBayesNameClassifier(train_data='name_gender.csv', train_split=.8)

# test is on the other 20% of name_gender.csv
#test_accuracy = trained_nbayes.test_nbayes_data()
#print('Test accuracy of trained_nbayes:', test_accuracy)





""" Test the already trained LSTM model on your custom data """
print('\nTesting the pre-trained LSTM model')

# model was trained on 80% of name_gender.csv
loaded_lstm = LSTMNameClassifier(model_to_load='lstm_name_gender_split80')

# try it on your custom data
#loaded_lstm.test_lstm_model(test_data='my_custom_data.csv')

# try out a name
loaded_lstm.test_lstm_model(names=['Andreas', 'Bernhard'])

""" Train and test LSTM from scratch """
# model is trained on 80% of name_gender.csv
#trained_lstm = LSTMNameClassifier(train_data='name_gender.csv', train_split=.8)
