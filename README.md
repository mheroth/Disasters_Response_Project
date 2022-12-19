# Disaster Response Project

Short text message classification algorithm

## Summary

In this project we use labeled text data provided by Figure Eight to train a machine learning model and perform a multioutput classification in a web app. The model is build by using a sklearn pipeline including a random forest classifier. Gridsearch is then used to tune the model parameters.

A user can enter text inside a Flask web app and get his message classified into multiple categories.

## Getting started

1. Run ETL pipeline that cleans data and stores in database
	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run ML pipeline that trains classifier and saves the model as a pickle file
	`python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl`
3. Run the following command in the app's directory to run the web app
	`python run.py`
4. Visit `http://0.0.0.0:3001/` to access the web app

## Repository

* `data/`: Holds the final ETL pipeline `process_data.py` and the development notebook. Also holds the database and the .csv files
* `model/`: Holds the final ML pipeline `train_classifier.py` and the development notebook. The pickled model is also stored here.
* `app/`: Flask web app `run.py` with templates
