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
    ''' load data from the database_filepath and gives the matrices X and Y for training and in addition the     category_names back 
    
    Parameters: 
    database_filepath: filepath of the database
    
    Returns:
    X: matrix of different input text which can used as training data
    Y: matrix of corresponding category (one out of 36) which can used as training data
    category_names: names of the 36 differentcategorys
    
    '''
    # create engine
    engine = create_engine('sqlite:///' + database_filepath)
    # read the sql table and save it in a pandas dataframe
    df = pd.read_sql_table('InsertTableName', 'sqlite:///' + database_filepath)
    # convert 'related' value of 2  to binary 1
    df.related.replace(2,1,inplace=True)
    # extract input text for later training
    X = df.message.values
    # extract corresponding categories
    Y = df[df.columns[4:]].values
    # get names of categories out of dataframe columns
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    ''' receives a text and gives tokenized list of relevant words back
    
    Parameters:
    text: text with punctuation, lower and upper cases as string
    
    Returns:
    tokens: words in a list without punctuation and only lower cases and without english stopwords
    
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # lemmatize and remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return tokens


def build_model():
    ''' build a pipeline with a CountVectorizer, TfidTransformer and a MultiOutputClassifier and create a         grid search object
    
    Returns:
    cv: GridSearchCV object using the above mentioned pipeline and the in this function defined parameters
    
    '''
    # Building pipeline with random forest classification
    pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10, n_jobs=-1)))])
    # define parameters for grid search
    parameters =  {
        'clf__estimator__n_estimators': [10, 15],
        'clf__estimator__min_samples_split': [2, 4]}
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' evaluate the model using the classification report and print them for each category name
    
    Parameters:
    model: the trained model
    X_text: test input dataset
    Y_test: text output dataset
    categroy_names: list of category names
    '''
    # calculate prediction for test input dataset
    y_pred_rf = model.predict(X_test)
    # get classification report for every category name
    for i in range(36):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred_rf[:,i]))


def save_model(model, model_filepath):
    ''' save the model using pickle
    
    Parameters:
    model: trained model
    model_filepath: filepath where the model should be stored
    
    '''
    # save model with pickle
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