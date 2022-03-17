from turtle import position
from flask import Flask, render_template, request
from sentiment_analysis import *
import pandas as pd
app = Flask(__name__)

class Tweet:
    def __init__(self, tweet, polarity, sentiment):
        self.tweet = str(tweet)
        self.polarity = float(polarity)
        self.sentiment = str(sentiment)
    def __repr__(self):
        return f'{self.tweet} -- {self.polarity}  -- {self.sentiment}'
        

@app.route('/')
def load_page():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def searching():
    if request.method == 'POST':
        search_word = request.form['searchinput']
        global positive, negative
        positive, negative = main(search_word)
        return render_template('sentiment.html', search_word = search_word.capitalize())    

@app.route('/about', methods=['GET', 'POST'])
def about_page():
    return render_template('about.html')

@app.route('/positive', methods=['GET', 'POST'])
def positive():
    allTweet = []
    for i in range(positive.shape[0]):
        tweet = Tweet(tweet = positive.iloc[i, 0], polarity=round(positive.iloc[i, 2], 2), sentiment=positive.iloc[i, 3])
        allTweet.append(tweet)
    return render_template('positive.html', allTweet = allTweet, text = "Positive")

@app.route('/negative', methods=['GET', 'POST'])
def negative():
    allTweet = []
    for i in range(negative.shape[0]):
        tweet = Tweet(tweet = negative.iloc[i, 0], polarity=round(negative.iloc[i, 2], 2), sentiment=negative.iloc[i, 3])
        allTweet.append(tweet)
    return render_template('positive.html', allTweet = allTweet, text = "Negative")

if __name__ == '__main__':
    app.run(debug = True)
