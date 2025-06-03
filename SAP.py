import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

plt.style.use('ggplot')

import nltk

#Read in the data
df = pd.read_csv('input/amazon-food-review/Reviews.csv')

#Display graphical representation of the first 5 rows
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
##plt.show()

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
print(vaders.head())