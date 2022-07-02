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
import os

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
features_car = ["", "engine", "wheels", "ev", "airbags", "average", "economy", "safety", "display", "infotainment",
            "cruise", "camera", "comfort", "brakes", "seats"]
features_bike = ["", 'engine', 'wheel', 'mileage', 'average', 'safety', 'speed']

features_phone = ['', 'battery', 'camera', 'display', 'processor', 'front camera', 'rear camera', 'weight']
features = []

def main(search_tag, type_prod):
    pd.set_option("display.max_colwidth", None)
    # SEARCH TAG, PRODUCT TYPE
    print(type_prod)
    if type_prod == 'car':
        features = features_car
    elif type_prod == 'bike':
        features = features_bike
    elif type_prod == 'phone':
        features = features_phone
    
    # CREATING MAIN DATAFRAME
    maindf = creating_data_frame(search_tag, features)
    print(maindf.shape)
    print(maindf.head)
    #Join with space that length is greater than 2, Here x is an argument
    maindf['Tweet'] = maindf['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    maindf['Tweet'] = maindf['Tweet'].apply(clean_up_tweet)
    maindf['Tweet'] = maindf['Tweet'].apply(remove_emojis)
    maindf['Tweet'] = maindf['Tweet'].apply(to_lower_case)
    maindf['Tweet'] = maindf['Tweet'].apply(cleaning_punctuations)
    maindf['Tweet'] = maindf['Tweet'].apply(remove_spaces)
    # Remove Empty Tweets from Main DataFrame
    maindf.drop(maindf[maindf['Tweet'] == ''].index, inplace=True)
    # Remove Duplicates Tweets from Main DataFrame
    maindf.drop_duplicates(subset="Tweet", keep='first', inplace=True)

    # Creating Three Columns of Subjectivity, Polarity, Sentiment and assign values by calling the getTweetSubjectivity(), getTweetPolarity(), getTextAnalysis()
    maindf['Subjectivity'] = maindf['Tweet'].apply(get_tweet_subjectivity)
    maindf['Polarity'] = maindf['Tweet'].apply(get_tweet_polarity)
    maindf['Sentiment'] = maindf['Polarity'].apply(get_tweet_analysis)

    # If we want to save all the tweets information into the csv file
    # maindf.to_csv('tweets1.csv')

    # Calculate percentage of positive, negative and neutral tweets
    positive_df = maindf[maindf['Sentiment'] == 'Positive']
    positive_percent = round(((positive_df.shape[0] / maindf.shape[0]) * 100), 2)
    
    negative_df = maindf[maindf['Sentiment'] == 'Negative']
    negative_percent = round(((negative_df.shape[0] / maindf.shape[0]) * 100), 2)

    neutral = maindf[maindf['Sentiment'] == 'Neutral']
    neutral_percent = round(((neutral.shape[0] / maindf.shape[0]) * 100), 2)

    # Creating the string of all tweets
    pos_tweet_str = create_str_tweets(positive_df)
    neg_tweet_str = create_str_tweets(negative_df)

    # Creating the tokens of all tweets string
    pos_tokens, neg_tokens = tokenize_str(pos_tweet_str, neg_tweet_str)

    # Finding the nouns from all tokens
    pos_nouns_lst = find_noun(pos_tokens)
    neg_nouns_lst = find_noun(neg_tokens)

    # Finding the frequency
    pos_freq_lst = find_frequency(pos_nouns_lst)
    neg_freq_lst = find_frequency(neg_nouns_lst)

    # If rating is not found then we return an empty list
    lst_features, lst_rating = find_percent_pos(pos_freq_lst, neg_freq_lst, features)
    if(len(lst_features) == 0 and len(lst_rating) == 0):
        print('not found')
        return [],[]

    #Creating a dataframe of features and their rating
    features_rating_df = create_df_features_rating(lst_features, lst_rating)
    
    # Plotting Pie Chart
    plotting_sentiment_pie_chart(positive_percent, negative_percent, neutral_percent)

    # Plotting Bar Graph
    plotting_bar_graph(features_rating_df)
    
    return positive_df, negative_df

# STAGE 1 - DATA GATHERING
# CREATING DATAFRAME
def creating_data_frame(search_word, features):
    # print(len(features))
    lst = []
    for feature in features:
        search_words = search_word + " " + feature 
        date_since = "2010-01-01"
        public_tweets = tweepy.Cursor(twitterApi.search_tweets, q=search_words, count=None, lang="en",
                                      since_id=date_since).items(100)
        df = pd.DataFrame(data=[tweet.text for tweet in public_tweets], columns=['Tweet'])
        lst.append(df)
    return pd.concat(lst, ignore_index=False)


# STAGE 2 - DATA CLEANING
# Remove Wildcards Characters
def clean_up_tweet(txt):
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


def to_lower_case(tweet):
    return str.lower(tweet)


def remove_spaces(tweet):
    return tweet.strip()


def cleaning_punctuations(text):
    english_punctuations = string.punctuation
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)


# STAGE 3 - GETTING SENTIMENTS USING TEXT BLOB AND PLOT THE BAR GRAPH AND PIE CHART
def get_tweet_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity


def get_tweet_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


def get_tweet_analysis(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"


# CREATING PIE GRAPH
def plotting_sentiment_pie_chart(positive_percent, negative_percent, neutral_percent):
    explode = (0, 0.1, 0)
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [positive_percent, negative_percent, neutral_percent]
    colors = ['yellowgreen', 'lightcoral', 'gold']
    plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=120)
    plt.legend(labels, loc=(-0.05, 0.05), shadow=True)
    plt.axis('equal')
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'static', 'img', 'SentimentAnalysis.png')
    plt.savefig(save_dir)
    plt.close()
    return


def create_str_tweets(sentiment_data_frame):
    sentiment_str = """"""
    sentiment_len = sentiment_data_frame.shape[0]
    for i in range(sentiment_len):
        sentiment_str += ' ' + sentiment_data_frame.iloc[i, 0]
    return sentiment_str


def tokenize_str(pos_str, neg_str):
    neg_tweet_token = word_tokenize(neg_str)
    pos_tweet_token = word_tokenize(pos_str)
    return pos_tweet_token, neg_tweet_token


def find_noun(tokens):
    noun_lst = []
    for token in tokens:
        if nltk.pos_tag([token])[0][1] == 'NN':
            noun_lst.append(nltk.pos_tag([token])[0][0])
    return noun_lst


def find_frequency(noun_list):
    fdist = FreqDist()
    for noun in noun_list:
        fdist[noun] += 1
    return fdist


def find_percent_pos(fdist_pos, fdist_neg, features):
    lst_features = []
    lst_rating = []
    for feature in features:
        if fdist_pos[feature] != 0:
            lst_features.append(feature)
            rating = (fdist_pos[feature] / (fdist_neg[feature] + fdist_pos[feature])) * 10
            lst_rating.append(int(rating))
    return lst_features, lst_rating


def create_df_features_rating(features, rating):
    df_positive = pd.DataFrame({'Features': features, 'Rating': rating})
    return df_positive


def plotting_bar_graph(df_pos):
    df_positive_plot = df_pos.nlargest(df_pos.shape[0], columns='Rating')
    sns_plot = sns.barplot(data=df_positive_plot, y='Features', x='Rating')
    sns.despine()
    fig = sns_plot.get_figure()
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'static', 'img', 'featuresBarGraph.png')
    fig.savefig(save_dir)
    plt.close(fig)

# This runs the code without web intergration
if __name__ == '__main__':
    main("realme 6", 'phone')