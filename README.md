# Fotball tweets analysis

This is a project developed as a final project at the Data Science bootcamp organised by Sages.

The project covers the analysis of tweets about football matches and is particularly focused on sentiment analysis. 

The purpose of this project is to assess whether it is possible to identify
important actions of the match, and what's more - its final result, 
based on sentiment and number of the tweets over the match time. 

The project consists of 9 main parts:
1. Football tweets scraping 
2. Preprocessing of training dataset - sentiment140 (which includes random tweets with sentiment labels)
3. Preprocessing of football tweets.
4. Comparison of different combinations of vectorizers (with various parameters) and classifiers parformed on train dataset.
5. Creating the classification model with best combination of vectorizers, classifiers and their parameters
6. Creating fasttext model.
7. Classification of football tweets - sentiment prediction.
8. Fotball tweets analyses - tweets count and sentiment over the match time, words' frequency etc.
9. Comparison of the result with course of the game.

### Data

Not listed project task was also resolving the problem of lack of training data - finding best training dataset with sentiment labels.
The *sentiment140* dataset was choosen which originated from Stanford University. This set contains tweets with sentiment labels
which were assigned automaticaly based on emoticons - positive or negative. 
More info on the dataset can be found from the link: http://help.sentiment140.com/for-students/ . 
The reduced version of this dataset was used in this project - 60K tweets and two sentiment classes: 0 - negative, 1 - positive.
Football tweets were scraped by using *tweepy* python package. 

### Data preprocessing

Preprocessing of the tweets includes:
- removing users names (@user_name), hashtags, punctuation, links and optionaly: stopwords, short words (with length < 2). 
- replacing slang words with their meaning, replacing double (or more) whitespaces with one space, replacing contractions with full expressions, replacing triple and more letters with doubles
- tokenization
- optional: stemming

## Project structure
- src - directory with files containing all scripts used for preprocessing, training the models, their comparisons, sentiment prediction and results visualisation
- utils - directory with files containing functions and constants which are used in all modules
- models - directory in which all the models are saved
- data - directory whith all tweets files and other data which are used as input of the project
- results - directory containg classifiers comparison files and other directories for each match where all result files (particularly graphs) are saved


## How to use

To run the project you need:
- Python 3.6
- pip
- Microsoft Visual C++ 14.0
- Anaconda

Open the command line and create conda environment:
```
conda create --name myenv
```
Activate the environment:
```
#windows
activate myenv

#linux
source activate myenv
```
Install all necessary packages:
```
pip install -r requirements.txt 
```
Run the jupyter notebook:
```
jupyter notebook
```
And enjoy the project!
