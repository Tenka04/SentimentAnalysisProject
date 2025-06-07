import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline

plt.style.use('ggplot')

import nltk

#Read in the data
df = pd.read_csv('input/amazon-food-review/Reviews.csv')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

test_sentence = "This is a gross product!"

#Run For Roberta

def polarity_scores(test_sentence):
    encoded_tweets = tokenizer(
        test_sentence,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    output = model(**encoded_tweets)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.head(500).iterrows(), total=500):
    try:
        text = row['Text']
        myid = row['Id']
        roberta_res = polarity_scores(text)
        res[myid] = {**roberta_res}
    except RuntimeError :
        print(f'Broken for {myid}, skipping...')

# Convert results to DataFrame and print first 5 rows
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
print(results_df.head())

results_df.columns

sns.pairplot(data=results_df, 
             vars=['roberta_neg', 'roberta_neu','roberta_pos'], 
             hue='Score',
             palette='tab10')
plt.show()