from flask import Flask, render_template, request
from sentiment_analysis import *
app = Flask(__name__)

class Tweet:
    def __init__(self, tweet, polarity, sentiment):
        self.tweet = str(tweet)
        self.polarity = float(polarity)
        self.sentiment = str(sentiment)
    def __repr__(self):
        return f'{self.tweet} -- {self.polarity}  -- {self.sentiment}'
        

@app.route('/')
def load_home_page():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def searching():
    if request.method == 'POST':
        search_word = request.form['searchinput']
        type_prod = request.form['chooseprod']
        global positive, negative

        #calling of main function
        positive, negative = main(search_word, type_prod)
        
        if type_prod == 'car':
            disp_string = search_word.capitalize() + ' ' + 'Car'
        elif type_prod == 'bike':
            disp_string = search_word.capitalize() + ' ' + 'Bike'
        elif type_prod == 'phone':
            disp_string = search_word.capitalize() + ' ' + 'Phone'
        
        template = 'sentiment.html'
        if len(positive) == 0 and len(negative) == 0:
            template = 'empty.html'
        return render_template(template, search_word = disp_string)    

@app.route('/about', methods=['GET', 'POST'])
def about_page():
    return render_template('about.html')

@app.route('/positive', methods=['GET', 'POST'])
def show_positive_tweet():
    allTweet = []
    for i in range(positive.shape[0]):
        tweet = Tweet(tweet = positive.iloc[i, 0], polarity=round(positive.iloc[i, 2], 2), sentiment=positive.iloc[i, 3])
        allTweet.append(tweet)
    return render_template('show_tweet.html', allTweet = allTweet, text = "Positive")

@app.route('/negative', methods=['GET', 'POST'])
def show_negative_tweet():
    allTweet = []
    for i in range(negative.shape[0]):
        tweet = Tweet(tweet = negative.iloc[i, 0], polarity=round(negative.iloc[i, 2], 2), sentiment=negative.iloc[i, 3])
        allTweet.append(tweet)
    return render_template('show_tweet.html', allTweet = allTweet, text = "Negative")

if __name__ == '__main__':
    app.run(debug = True)
