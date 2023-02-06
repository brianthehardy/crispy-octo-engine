import pandas
import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', 5)


'''
Column 1: the ID of the statement ([ID].json).
Column 2: the label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).
'''
df_headers = [ 'idx', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
# Import dataset
df_train = pd.read_csv("Datasets/train.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]
df_test = pd.read_csv("Datasets/test.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]
df_validate = pd.read_csv("Datasets/valid.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]

all_data = {'train': df_train, 'test': df_test, 'val': df_validate}
vectorized_data = {}


# Hard coding to get this to work
x_train = all_data['train']['statement'].to_numpy()
y_train = all_data['train']['label'].to_numpy()


# Data cleaning after import
for df in all_data:
    # Replace the labels with numerical representations, 0-5
    all_data[df]['label'] = all_data[df].label.map({'pants-fire':0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5})

    # lowercase, and remove punctuation
    all_data[df]['statement'] = all_data[df].statement.map(lambda x: x.lower()).str.replace('[^\w\s]', '')

    # DataFlair - Initialize a TfidfVectorizer
    # Term Frequency - Inverse Document Frequency
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform train set, transform test set
    idx = df+'_x'

    # Turn the statement into a vector
    vectorized_data[idx] = tfidf_vectorizer.fit_transform( all_data[df]['statement'].to_numpy() )
    vectorized_data[idx] = tfidf_vectorizer.fit_transform(all_data[df]['statement'].to_numpy())
    print(df, "/", idx, ": ", vectorized_data[idx])
    tfidf_train = tfidf_vectorizer.fit_transform( vectorized_data[idx] )
    # tfidf_test = tfidf_vectorizer.transform(vectorized_data[idx])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # First, let's just do a blind classifier with no cleaning
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB().fit(all_data['train']['statement'], all_data['train']['label'])

    predicted = model.predict(all_data['test']['statement'])
    print(np.mean(predicted == all_data['test']['label']))