# Tweet Sentiment Classifier

Our goal is to train a model on a dataset of tweets that are labeled as either negative, neutral, or positive. Depending on the context of the text, the sentiment will be classified as one of the 3 options. We originally had planned to make a leaderboard competition, but due to complications with staff we were unable to obtain LiveDataLab to perform these competitions. Instead we will just develop the model and provide the sentiment of tweets we analyzed and trained our model on.

### | [Video](https://drive.google.com/file/d/1KMEMokQWhzdSIaEsEW94WgDUUzUXkIWW/view?usp=sharing) | [Documentation and Final Report](https://github.com/Raove/BABL_CS410/blob/main/Documentation%20and%20Final%20Report.pdf) | [Code](https://github.com/Raove/BABL_CS410/blob/main/tweet_classifier_model.py) |


### Project set up

For you to be able to run this project you will need to have completed MP3.2 as this is a similar process to that. You will need the following installed.
    
    1. pip install transformers
    2. pip install torch
    3. pip install pandas

## Process

So we are training our model with the dataset and testing it with the same dataset in the follwoing way. Here I will show our dataset and explain right after.

| text | selected_text | sentiment |
| - | - | - |
| my boss is bullying me... | bullying me | negative |
| I`d have responded, if I were going | I`d have responded, if I were going | neutral |
| 2am feedings for the baby are fun when he is all smiles and coos | fun | positive |

So the plan is to train the model the selected text to decipher the tweets sentiments, then after the model has been trained, we will test our model with the text from the tweets and compare them to the actual sentiment to find out the model's accuracy and score. This is where we were going to have our leaderboard competition on LiveDataLab, but we were unable to obtain credentials from staff to perform these tasks.