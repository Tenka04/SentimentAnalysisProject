import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

plt.style.use('ggplot')

import nltk

#Read in the data
df = pd.read_csv('input/amazon-food-review/Reviews.csv')

#Display graphical representation of the first 5 rows
#ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by stars', figsize=(10, 5))
#ax.set_xlabel('Review Stars')
#plt.show()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

#run the polarity scores on the reviews
res = {}
for i, row in tqdm(df.head(500).iterrows(), total=500):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders= pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

#seniment score and meta data
#print(vaders.head())

#PLOT VADER RESULTS
ax = sns.barplot(data=vaders, y='pos', x='Score')
ax.set_title('Vader Compound Score by Review Stars')
#plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, y='pos', x='Score', ax=ax[0])
sns.barplot(data=vaders, y='neu', x='Score', ax=ax[1])
sns.barplot(data=vaders, y='neg', x='Score', ax=ax[2])
ax[0].set_title('Positive')
ax[1].set_title('Neutral')
ax[2].set_title('Negative')
#plt.show()

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

test_sentence = "This is a gross product!"

#Run For Roberta
encoded_tweets = tokenizer(test_sentence ,return_tensors='pt')
output = model(**encoded_tweets)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(f"Roberta scores: {scores_dict}")

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
        vader_res = sia.polarity_scores(text)
        vader_res_rename = {}
        for key, value in vader_res.items():
            vader_res_rename[f'vader_{key}'] = value
        roberta_res = polarity_scores(text)
        res[myid] = {**vader_res_rename, **roberta_res}
    except RuntimeError :
        print(f'Broken for {myid}, skipping...')

# Convert results to DataFrame and print first 5 rows
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
print(results_df.head())

results_df.columns

sns.pairplot(data=results_df, 
             vars=['vader_pos', 'vader_neg', 'vader_neu', 
                   'roberta_pos', 'roberta_neg', 'roberta_neu'], 
             hue='Score',
             palette='tab10')
plt.show()