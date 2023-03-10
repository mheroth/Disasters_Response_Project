import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' load messages from messages_filepath and categories from categories_filepath and give them together as     one pandas dataframe and a list of the category names back
    
    Parameters: 
    message_filepath: filepath of messages data file
    categories_filepath: filepath of categories data file
    
    '''
    # read messages
    messages = pd.read_csv(messages_filepath, dtype=str)
    # read categories
    categories = pd.read_csv(categories_filepath, dtype=str)
    # merge messages and categories together
    df = messages.merge(categories, how='left', on=['id'])
    # split categories with ';' (extract different categories out of a string)
    categories = df['categories'].str.split(';', expand=True)
    # get row
    row = categories.iloc[:1]
    # initialize category_colnames
    category_colnames=[]
    # get category_colnames out of row
    category_colnames = list(row.apply(lambda x: x[0][:-2], axis=0))
    # set categories columns with new created category_colnames
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # convert related column to binary
    categories.loc[categories["related"] == 2] = 1
    #categories.drop(categories[categories['related']==2].index,inplace=True)
    #print(categories.related.unique())
    # drop old categories column
    df.drop('categories', axis=1, inplace=True)
    # concat df with new categories
    df = pd.concat([df, categories], axis=1)
    return df

def clean_data(df):
    ''' receive a dataframe and give the cleaned dataframe with no duplicates back
    
    Parameters:
    df: dataframe with duplicates
    
    Returns:
    df: dataframe without duplicates
    
    '''
    df = df.drop_duplicates()
    #print(df)
    return df


def save_data(df, database_filename):
    ''' saves a dataframe in a database
    
    Parameters
    df: dataframe
    database_filename: filename of the target database
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()