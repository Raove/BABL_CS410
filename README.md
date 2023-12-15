# Tweet Sentiment Classifier

Our goal is to train a model on a dataset of tweets that are labeled as either negative, neutral, or positive. Depending on the context of the text, the sentiment will be classified as one of the 3 options. We originally had planned to make a leaderboard competition, but due to complications with staff we were unable to obtain LiveDataLab credentials and Microsoft Azure credits needed to integrate this work with LiveDataLab.  Instead we developed the model, provided the sentiment of tweets we analyzed and trained our model on, and illustrated that this code could be augmented by students in a competition on LiveDataLab similarly to past Programming Assignments in this class.

### | [Video](https://drive.google.com/file/d/1KMEMokQWhzdSIaEsEW94WgDUUzUXkIWW/view?usp=sharing) | [Documentation and Final Report](https://github.com/Raove/BABL_CS410/blob/main/Documentation%20and%20Final%20Report.pdf) | [Code](https://github.com/Raove/BABL_CS410/blob/main/tweet_classifier_model.py) |


### Project set up

To run this project, first clone this repository. This should include the archive folder which contains the Tweets.csv dataset required for model training.

Then navigate to the tweet_classifier_model.py file. Ensure the following are installed:
    
    1. pip install transformers
    2. pip install torch
    3. pip install pandas

Finally, perform the python3 tweet_classifier_model.py command to run the file. Make edits to the hyperparameters and model form as needed to improve performance.

## Process

So we are training our model with the dataset and testing it with the same dataset in the follwoing way. Here is a sample of the dataset used.

| text | selected_text | sentiment |
| - | - | - |
| my boss is bullying me... | bullying me | negative |
| I`d have responded, if I were going | I`d have responded, if I were going | neutral |
| 2am feedings for the baby are fun when he is all smiles and coos | fun | positive |

The model is trained on the selected text to decipher the tweet's sentiment, then after the model has been trained, we will test our model with the text from the tweets and compare them to the actual sentiment to find out the model's accuracy. At this stage we would integrate the baseline model with LiveDataLab for students to iterate on in a leaderboard competition, but we were unable to obtain credentials from staff to perform these tasks. For more information, see the Documentation and Final Report file linked above.
