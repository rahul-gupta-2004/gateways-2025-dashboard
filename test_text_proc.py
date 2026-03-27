import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)
df = pd.read_csv('C5-FestDataset - fest_dataset.csv')
text = " ".join(df['Feedback on Fest'].dropna().astype(str))
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
print("Wordcloud generated.")

sia = SentimentIntensityAnalyzer()
df['score'] = df['Feedback on Fest'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['sentiment'] = df['score'].apply(lambda s: 'Pos' if s > 0.05 else ('Neg' if s < -0.05 else 'Neu'))
print(df['sentiment'].value_counts())
