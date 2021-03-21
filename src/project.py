import matplotlib.pyplot as plt
import pandas as pd
import csv
import re

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from numpy import concatenate
import pyLDAvis
import pyLDAvis.sklearn

nltk.download('wordnet')
nltk.download('stopwords')


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
        self.message_label4.grid(row=8, column=1)
        self.lda_button.grid(row=9, column=2,
                             padx=10, pady=15)

    """ 
 
     DATA CLEANING
     
     1. 
     
    """

    def clean_csv(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            col_list = ["text"]
            array = []
            df = pd.read_csv(self.file_name, usecols=col_list)
            stopwords = nltk.corpus.stopwords.words('english')
            newStopWords = ['amp', 'rt', 'https', 'http']
            stopwords.append(newStopWords)
            lemmatizer = WordNetLemmatizer()
            ps = PorterStemmer()

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
                    # clean_tweets.append(set(tweet).difference(stop_words))
                    filtered_sentence = [w for w in tweet if not w in stopwords]

                    for word in filtered_sentence:
                        test = lemmatizer.lemmatize(word, wordnet.VERB)
                        nltk_tokens = nltk.word_tokenize(test)
                    array.append(nltk_tokens)

            # Create a new dataframe of the cleaned dataset
            df['text'] = array
            # Add to file called clean_csv
            clean_file = r"../JSON-CSV-files/clean_file_LDA.csv"
            df.to_csv(clean_file, index=False)
            msg.showinfo('File', 'File has been cleaned. Ready to be converted.')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    """
         DATA PROCESSING
            
         1. Build a dictionary with all the words in the dataset
         2. Sort the word counts (using the dictionary created) of each tweet in a corpus.
            
          Algorithm used is the Latent-Dirichlet Allocation, a topic modelling algorithm which learns the latent 
          topics of a group of documents, ideal for unlabeled data like that from a twitter dataset.
            
         # https://ourcodingclub.github.io/tutorials/topic-modelling-python/
            
    """

    def lda(self):
        try:
            # https://ourcodingclub.github.io/tutorials/topic-modelling-python/
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            df = pd.read_csv(self.file_name)

            """
                Functions for making a vocabulary of all the words of the twitter data
                
                1. Use the CountVectoriser to create a Document-Term matrix
                2. We specify to only include words that appear in less than 80% of and appear in at least 2 documents
            """

            # https://stackabuse.com/python-for-nlp-topic-modeling/
            count_vect = CountVectorizer(max_df=0.8, min_df=2)
            doc_term_matrix = count_vect.fit_transform(df["text"].values.astype('U'))

            """
                Latent Dirilchet Allocation use
                
                1. Include the number of topics we want to create
                2. Create a Visualisation in pyLDAvis
            """
            # Create 10 Topics
            number_of_topics = 10

            LDA = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', random_state=42,
                                            n_jobs=-1)
            LDA_output = LDA.fit(doc_term_matrix)

            visualisation = pyLDAvis.sklearn.prepare(LDA_output, doc_term_matrix, count_vect, mds='tsne')
            # visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)


# Driver Code
root = Tk()
root.title('Twitter Visualisation Tool')

obj = TwitterApplication(root)
root.geometry('600x300')
root.mainloop()