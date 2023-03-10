# Disaster Response Project

## Installation
There should be the following libraries installed to run the code here beyond the Anaconda distribution of Python: 
- sys
- pandas
- sqlalchemy
- pickle
- sklearn.model_selection
- nltk
- json
- plotly
- flask

The code should run with no issues using Python versions 3.*.

Short text message classification algorithm

## Project Motivation

In this project we use labeled text data provided by Figure Eight to train a machine learning model and perform a multioutput classification in a web app. The model is build by using a sklearn pipeline including a random forest classifier. Gridsearch is then used to tune the model parameters. 
This app can be used by several disaster response agencies. This will help the disaster victims to receive prompt medical aid and speedy recovery from the effects of the disasters.

A user can enter text inside a Flask web app and get his message classified into multiple categories.

## Getting started

1. Run ETL pipeline that cleans data and stores in database
	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run ML pipeline that trains classifier and saves the model as a pickle file
	`python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl`
3. Run the following command in the app's directory to run the web app
	`python run.py`
4. Visit `http://0.0.0.0:3001/` to access the web app

## File Description

* `data/`: Holds the final ETL pipeline `process_data.py` and the development notebook. Also holds the database and the .csv files
* `model/`: Holds the final ML pipeline `train_classifier.py` and the development notebook. The pickled model is also stored here.
* `app/`: Flask web app `run.py` with templates

Structure of the project:
app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py

|- classifier.pkl # saved model

README.md

## Results
The code and the results can be used on the following web app:  `http://0.0.0.0:3001/`
Web App visualisation:
![WebApp](https://user-images.githubusercontent.com/95216325/208512883-cc996c62-52a1-4f1e-bccb-7662a483d9a3.PNG)


## Licensing, Authors, Acknowledgements
In this project the disaster data from Appen (formally Figure 8) is used. For more information about the Licensing of the data or descroptive information visit the following homepage: https://appen.com/. Otherwise, feel free to use the code here as you would like!

