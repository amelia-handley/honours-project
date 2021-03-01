import matplotlib
import pandas as pd
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg
import json, csv
import re, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from spacy.lang.en.tokenizer_exceptions import word

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()


class TwitterApplication:

    def __init__(self, root):

        """ Code to Create GUI in Python was taken from:
        https://www.geeksforgeeks.org/create-a-gui-to-convert-csv-file-into-excel-file-using-python/
        Last accessed: 28/01/2021
        """

        self.root = root
        self.filename = ''
        self.f = Frame(self.root, height=200, width=300)

        # Place the frame on root window
        self.f.pack()

        # Creating label widgets
        self.message_label = Label(self.f,
                                   text='Twitter Visualisation Tool',
                                   font=('Arial', 19, 'bold'))
        self.message_label2 = Label(self.f,
                                    text='Clean .csv file.',
                                    font=('Arial', 12, 'underline'))
        """
        self.message_label3 = Label(self.f,
                                    text='Convert .csv file to .json.',
                                    font=('Arial', 12, 'underline'))
        """
        self.message_label4 = Label(self.f,
                                    text='Algorithms',
                                    font=('Arial', 12, 'underline'))

        # Buttons
        self.clean_button = Button(self.f,
                                   text='Clean',
                                   font=('Arial', 14),
                                   bg='Red',
                                   fg='Black',
                                   command=self.clean_csv)

        """
        self.convert_button = Button(self.f,
                                     text='Convert',
                                     font=('Arial', 14),
                                     bg='Orange',
                                     fg='Black',
                                     command=self.convert_csv_to_json)
        """

        self.k_means_button = Button(self.f,
                                     text='K-Means',
                                     font=('Arial', 14),
                                     bg='Green',
                                     fg='Black',
                                     command=self.k_means)

        self.lda_button = Button(self.f,
                                 text='LDA',
                                 font=('Arial', 14),
                                 bg='Blue',
                                 fg='Black',
                                 command=self.lda)

        # Placing the widgets using grid manager
        self.message_label.grid(row=0, column=1)
        self.message_label2.grid(row=2, column=1)
        self.clean_button.grid(row=4, column=1,
                               padx=0, pady=15)
        """
        self.message_label3.grid(row=5, column=1)
        self.convert_button.grid(row=6, column=1,
                                 padx=0, pady=15)
        """
        self.message_label4.grid(row=8, column=1)
        self.k_means_button.grid(row=9, column=0,
                                 padx=10, pady=15)
        self.lda_button.grid(row=9, column=2,
                             padx=10, pady=15)

    def lemmatize_text(self):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in self.split()]

    """
    def remove_links(self):
        '''Takes a string and removes web links from it'''
        self = re.sub(r'http\S+', '', self)  # remove http links
        self = re.sub(r'bit.ly/\S+', '', self)  # rempve bitly links
        self = self.strip('[link]')  # remove [links]`
        return self

    def remove_users(self):
        # Takes a string and removes retweet and @user information
        self = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', " ", self)  # remove retweet
        self = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', self)  # remove tweeted at
        return self

    # cleaning master function
    def clean_tweet(self, bigrams=False):
        self = self.remove_users(self)
        self = self.remove_links(self)
        self = self.lower()  # lower case
        self = re.sub('[' + my_punctuation + ']+', ' ', self)  # strip punctuation
        self = re.sub('\s+', ' ', self)  # remove double spacing
        self = re.sub('([0-9]+)', '', self)  # remove numbers
        tweet_token_list = [word for word in self.split(' ')
                            if word not in my_stopwords]  # remove stopwords

        tweet_token_list = [word_rooter(word) if '#' not in word else word
                            for word in tweet_token_list]  # apply word rooter
        if bigrams:
            tweet_token_list = tweet_token_list + [tweet_token_list[i] + '_' + tweet_token_list[i + 1]
                                                   for i in range(len(tweet_token_list) - 1)]
        self = ' '.join(tweet_token_list)
        return self
        """

    def clean_csv(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            col_list = ["text"]
            clean_stop_words_tweets = []
            clean_lemmatizer_tweets = []
            df = pd.read_csv(self.file_name, usecols=col_list)
            stop_words = set(stopwords.words('english'))
            #newStopWords = ('amp', 'rt')
           # stopwords.append(newStopWords)

            if len(df) == 0:
                msg.showinfo('No Rows Selected', 'CSV file has no data.')
            else:
                for tweet in df["text"]:
                    """ Code taken to clean tweet data was taken from:
                    https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/analyze-tweet-sentiment-in-python/
                    Last accessed: 03/02/2021
                    
                    To remove URLs and non-alphanumeric characters
                    """

                    tweet = (re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",
                                    tweet.lower()).split())
                    #clean_stop_words_tweets.append(set(tweet).difference(stop_words))
                    filtered_sentence = [w for w in tweet if not w in stop_words]
                    #print(filtered_sentence)
                    for word in filtered_sentence:
                        clean_lemmatizer_tweets = (lemmatizer.lemmatize(word))
                        print(clean_lemmatizer_tweets)

                # Create a new dataframe of the cleaned dataset
                df['text'] = clean_lemmatizer_tweets
                # Add to file called clean_csv
                df.to_csv("../JSON-CSV-files/clean_csv.csv", index=False)
                msg.showinfo('File', 'File has been cleaned. Ready to be converted.')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    """
    def convert_csv_to_json(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))
            clean_tweets = []
            col_list = ["text"]
            df = pd.read_csv(self.file_name, usecols=col_list)

            # Next - Pandas DF to Excel file on disk
            if len(df) == 0:
                msg.showinfo('No Rows Selected', 'CSV is empty.')
            else:
                createjson = df.to_json()
                # print(createjson)
                with open("../JSON-CSV-files/TwitterDataJson.json", 'r') as write_file:
                    json.dump(createjson, write_file)
                msg.showinfo('JSON file created', 'JSON file created')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    def tokenize_and_stem(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    """

    """
        Using the Sklearn library to create the K-Means algorithm
    """
    def k_means(self):
        try:
            # https://github.com/Ashwanikumarkashyap/k-means-clustering-tweets-from-scratch/blob/master/main.py
            # Allow user to find the CSV file they wish to apply K-Means to
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))
            # Only look at the "text" column in the file
            col_list = ["text"]
            # Read CSV file
            df = pd.read_csv(self.file_name, usecols=col_list)

            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(df)

            true_k = 5  # Since there is only one sample
            model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
            model.fit(X)

            # https://pythonprogramminglanguage.com/kmeans-text-clustering/

            print("Top terms per cluster: ")
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            for i in range(true_k):
                print("Cluster %d:" % i),
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind])
                print()

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    """
        Latent-Dirichlet Allocation using the Sklearn library
    """
    def lda(self):
        try:
            # https://ourcodingclub.github.io/tutorials/topic-modelling-python/
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            col_list = ["text"]
            df = pd.read_csv(self.file_name, usecols=col_list)

            # count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
            count_vect = CountVectorizer(max_df=0.8, min_df=2)
            doc_term_matrix = count_vect.fit_transform(df['text'].values.astype('U'))

            LDA = LatentDirichletAllocation(n_components=5, random_state=42)
            LDA.fit(doc_term_matrix)

           # for i in range(10):
                # random_id = random.randint(0,len(count_vect.get_feature_names()))
                # print(count_vect.get_feature_names()[random_id])
            #first_topic = LDA.components_[0]
            #top_topic_words = first_topic.argsort()[-10:]
            # for i in top_topic_words:
            #   print(count_vect.get_feature_names()[i])

            for i,topic in enumerate(LDA.components_):
                print(f"Top 10 words for topic #{i}")
                print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
                print("\n")

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)


# Driver Code
root = Tk()
root.title('Twitter Visualisation Tool')

obj = TwitterApplication(root)
root.geometry('600x600')
root.mainloop()
