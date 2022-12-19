import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', 'sqlite:///' + database_filepath)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    # Todo: normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Todo: tokenize text
    words = word_tokenize(text)
    
    # Todo: lemmatize and remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return tokens


def build_model():
    # Random Forest Classification
    pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10, n_jobs=-1)))])
    parameters =  {
        'clf__estimator__n_estimators': [10, 15],
        'clf__estimator__min_samples_split': [2, 4]}
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_rf = model.predict(X_test)

    for i in range(36):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred_rf[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()