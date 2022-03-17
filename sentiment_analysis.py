from audioop import minmax
import tweepy
from textblob import TextBlob
import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize twitter account credentials
consumer_key = "82ZBWN7VF93E2PxiduNJxFt5b"
consumer_secret = "1BdNAkCIqmsUPCvc0fZH9yVl94Im7W26JSRgzxUgC3p0uM09Yh"
access_token = "3189799836-LN5NMDrrsoF8kfdgM4Np6IRADIO8bow7jgUuLtf"
access_token_secret = "TzDR6idvO6SkTrbB5lkf83e8hPv7kUuYRsq7X3UkIM5Mc"

# connect with twitter API by tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterApi = tweepy.API(auth, wait_on_rate_limit=True)

# LIST OF FEATURES
features = ["", "engine", "wheels", "ev", "airbags", "average", "economy", "safety", "display", "infotainment",
            "cruise", "camera", "comfort", "brakes", "seats"]
features1 = [""]

def main(searchTag):
    pd.set_option("display.max_colwidth", None)
    # SEARCH TAG
    searchtag = searchTag
    # CREATING MAIN DATAFRAME
    global maindf
    maindf = creatingDataFrame(searchtag, features)
    maindf['Tweet'] = maindf['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    maindf['Tweet'] = maindf['Tweet'].apply(cleanUpTweet)
    maindf['Tweet'] = maindf['Tweet'].apply(remove_emojis)
    maindf['Tweet'] = maindf['Tweet'].apply(to_lower_case)
    maindf['Tweet'] = maindf['Tweet'].apply(cleaning_punctuations)
    maindf['Tweet'] = maindf['Tweet'].apply(remove_spaces)
    maindf.drop(maindf[maindf['Tweet'] == ''].index, inplace=True)
    maindf.drop_duplicates(subset="Tweet", keep='first', inplace=True)
    maindf['Subjectivity'] = maindf['Tweet'].apply(getTweetSubjectivity)
    maindf['Polarity'] = maindf['Tweet'].apply(getTweetPolarity)
    maindf['Sentiment'] = maindf['Polarity'].apply(getTextAnalysis)
    # maindf.to_csv('tweets1.csv')

    global pos
    positive = maindf[maindf['Sentiment'] == 'Positive']
    pos = round(((positive.shape[0] / maindf.shape[0]) * 100), 2)

    global neg
    negative = maindf[maindf['Sentiment'] == 'Negative']
    neg = round(((negative.shape[0] / maindf.shape[0]) * 100), 2)

    global neut
    neutral = maindf[maindf['Sentiment'] == 'Neutral']
    neut = round(((neutral.shape[0] / maindf.shape[0]) * 100), 2)

    # print(f"{pos} % of positive")
    # print(f"{neg} % of negative")
    # print(f"{neut} % of neutral")

    pos_str = create_str(positive)
    neg_str = create_str(negative)

    # print(pos_str, "\n", neg_str)

    pos_tokens, neg_tokens = tokenizestr(pos_str, neg_str)
    # print(pos_tokens, "\n", neg_tokens)

    pos_nouns = findnoun(pos_tokens)
    neg_nouns = findnoun(neg_tokens)

    # print(pos_nouns, "\n", neg_nouns)
    # print(len(pos_nouns), "\n", len(neg_nouns))

    pos_freq_lst = findFrequency(pos_nouns)
    neg_freq_lst = findFrequency(neg_nouns)

    # print(type(pos_freq_lst), "\n", neg_freq_lst)

    lst_features, lst_rating = find_percent_pos(pos_freq_lst, neg_freq_lst)

    # print(lst_features, lst_rating)

    df_pos = create_DF_features_rating(lst_features, lst_rating)
    # print(df_pos.head())
    plottingSentimentPieChart()
    plottingBarGraph(df_pos)
    # print(positive.head())
    return positive, negative

# CREATING DATAFRAME
def creatingDataFrame(search, features):
    # print(len(features))
    lst = []
    for feature in features:
        search_words = search + " " + feature  # enter your words
        date_since = "2010-01-01"
        public_tweets = tweepy.Cursor(twitterApi.search_tweets, q=search_words, count=None, lang="en",
                                      since_id=date_since).items(100)
        df = pd.DataFrame(data=[[tweet.text] for tweet in public_tweets], columns=['Tweet'])
        lst.append(df)
    return pd.concat(lst, ignore_index=False)


# STAGE 1 - DATA CLEANING
def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    txt = re.sub("#[A-Za-z0-9_]+", "", txt)
    txt = re.sub("^\\s+|\\s+$", "", txt)
    txt = re.sub(r'RT : ', '', txt)
    txt = re.sub(r'&amp;', '', txt)
    txt = re.sub(r'rt,', '', txt)
    txt = re.sub(r'&gt;', '', txt)
    txt = re.sub(r'(.)1+', r'1', txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    txt = re.sub('[0-9]+', '', txt)
    txt = re.sub(r'\'', '', txt)
    txt = re.sub(r'\n', '', txt)
    txt = re.sub(r'“', '', txt)
    txt = re.sub(r'”', '', txt)
    txt = re.sub("[ \t]{2,}", " ", txt)
    return txt


def remove_emojis(tweet):
    emoji = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002500-\U00002BEF"  # chinese char
                       u"\U00002702-\U000027B0"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U00010000-\U0010ffff"
                       u"\u2640-\u2642"
                       u"\u2600-\u2B55"
                       u"\u200d"
                       u"\u23cf"
                       u"\u23e9"
                       u"\u231a"
                       u"\ufe0f"  # dingbats
                       u"\u3030"
                       "]+", re.UNICODE)
    return re.sub(emoji, '', tweet)


def to_lower_case(txt):
    return str.lower(txt)


def remove_spaces(txt):
    return txt.strip()


def cleaning_punctuations(text):
    english_punctuations = string.punctuation
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)


# STAGE 2 - GETTING SENTIMENTS USING TEXT BLOB
def getTweetSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity


def getTweetPolarity(txt):
    return TextBlob(txt).sentiment.polarity


# maindf['Subjectivity'] = maindf['Tweet'].apply(getTweetSubjectivity)
# maindf['Polarity'] = maindf['Tweet'].apply(getTweetPolarity)


def getTextAnalysis(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"


# CREATING PIE GRAPH
def plottingSentimentPieChart():
    explode = (0, 0.1, 0)
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [pos, neg, neut]
    colors = ['yellowgreen', 'lightcoral', 'gold']
    plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
    plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
    plt.axis('equal')
    plt.savefig("C:\\Users\\anant\\OneDrive\\Desktop\data\\Project\\Social Media Sentiment Analysis\\static\img\\SentimentAnalysis.png")
    plt.close()
    return


def create_str(sentimentdata):
    sentiment_str = """"""
    sentiment_len = sentimentdata.shape[0]
    for i in range(sentiment_len):
        # print(positive.iloc[i,0])
        sentiment_str += ' ' + sentimentdata.iloc[i, 0]
    return sentiment_str


def tokenizestr(pos_str, neg_str):
    neg_text_token = word_tokenize(neg_str)
    pos_text_token = word_tokenize(pos_str)
    return pos_text_token, neg_text_token


def findnoun(tokens):
    noun_lst = []
    for token in tokens:
        if nltk.pos_tag([token])[0][1] == 'NN':
            noun_lst.append(nltk.pos_tag([token])[0][0])
    return noun_lst


def findFrequency(nounList):
    fdist = FreqDist()
    for noun in nounList:
        fdist[noun] += 1
    return fdist


def find_percent_pos(fdist_pos, fdist_neg):
    lst_features = []
    lst_rating = []
    for feature in features:
        if fdist_pos[feature] != 0:
            lst_features.append(feature)
            rating = (fdist_pos[feature] / (fdist_neg[feature] + fdist_pos[feature])) * 10
            lst_rating.append(int(rating))
    # print(feature, " -> ", int(rating), " rating")
    return lst_features, lst_rating


def create_DF_features_rating(features, rating):
    df_positive = pd.DataFrame({'Features': features, 'Rating': rating})
    return df_positive


def plottingBarGraph(df_pos):
    df_positive_plot = df_pos.nlargest(df_pos.shape[0], columns='Rating')
    sns_plot = sns.barplot(data=df_positive_plot, y='Features', x='Rating')
    sns.despine()
    fig = sns_plot.get_figure()
    fig.savefig("C:\\Users\\anant\\OneDrive\\Desktop\data\\Project\\Social Media Sentiment Analysis\\static\img\\featuresBarGraph.png")
    plt.close(fig)

if __name__ == '__main__':
    main("BMW")