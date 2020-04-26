import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

#Step 1.1: Understanding our dataset
df = pd.read_table('SMSSpamCollection', sep='\t', header= None, names=['label', 'sms_message'] )

#Step 1.2: Data Preprocessing
df['label'] = df.label.map({'ham': 0, 'spam':1})

#Step 2.1: Training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state = 1)

#Step 2.2: Applying Bag of Words processing to our dataset.
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

#step 3: Naive Bayes implementation using scikit-learn
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

prediction = naive_bayes.predict(testing_data)

precision = precision_score(y_test, prediction)
accuracy = accuracy_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)



