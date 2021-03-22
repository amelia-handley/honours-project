import matplotlib.pyplot as plt
import pandas as pd
import re

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg

import nltk
from gensim import models
from gensim.matutils import corpus2dense
from nltk.stem import WordNetLemmatizer, PorterStemmer, wordnet
from numpy import concatenate
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn

nltk.download('wordnet')
nltk.download('stopwords')

"""
    Visualisation guided analysis (Functions)
    
    After finding a topic of interest from the pyLDAvis visualisation
    1. Sort the data based on this topic for further information.
    2. Translate the corpus to match a Pandas dataframe
"""


def translateLdaIdx(myLdaModel, myLdaViz):
    ldaVizIdx = myLdaViz[0].index
    return list(ldaVizIdx)


def createDenseMat(myLdaModel, newIdx):
    numTopics = myLdaModel.n_components
    myDense = corpus2dense(myLdaModel, numTopics)
    myDf = pd.DataFrame(myDense)
    # mySortedDf = myDf.transpose()
    mySortedDf = myDf.transpose()[newIdx]
    mySortedDf.columns = ['topic' + str(i + 1) for i in range(numTopics)]
    return mySortedDf


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
        self.lda_button.grid(row=9, column=0,
                             padx=10, pady=15)
        self.lda_button.grid(row=9, column=2,
                             padx=10, pady=15)

    """ 
 
     DATA CLEANING
     
     Cleaning the data is one of the main challenges of data analysis. Simply splitting the words with spaces between
     them is insufficient. Therefore, the ntlk package is used to address each of the data cleaning strategies
     
     1. word capitalisation 
     2. punctuation
     3. singular-plural versions of same word (lemmisation)
     4. common words like 'and' (stopwords)
     
    """

    def clean_csv(self):
        try:
            # Find the .csv file needing to be cleaned
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            # Filter the .csv file to find the text column to retrieve the tweet data
            col_list = ["text"]

            df = pd.read_csv(self.file_name, usecols=col_list)
            # Stop words which have been added to the list include 'amp', 'rt', 'https'
            stopwords = nltk.corpus.stopwords.words('english')
            newStopWords = ['amp', 'rt', 'https', 'http']
            stopwords.append(newStopWords)
            # Array list to store the cleaned tweets
            clean_data = []

            # Initialise lemmatiser and stemmer
            lemmatizer = WordNetLemmatizer()
            ps = PorterStemmer()

            # If the file is empty, display a message
            if len(df) == 0:
                msg.showinfo('No Rows Selected', 'CSV file has no data.')
            else:
                for tweet in df["text"]:
                    """
                    To remove URLs and non-alphanumeric characters
                    """
                    tweet = (re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",
                                    tweet.lower()).split())
                    # clean_tweets.append(set(tweet).difference(stop_words))

                    """
                        Remove Stop Words from the tweets
                    """
                    filtered_sentence = [w for w in tweet if not w in stopwords]
                    clean_data.append(filtered_sentence)

                    """
                        For each word in each tweet, perform lemmatisation and tokenisation
                    """

                    for word in filtered_sentence:
                        test = lemmatizer.lemmatize(word, wordnet.VERB)
                        nltk_tokens = nltk.word_tokenize(test)
                    clean_data.append(nltk_tokens)

            # Create a new dataframe of the cleaned dataset
            df['text'] = clean_data
            # Add to file called clean_csv
            clean_file = r"../JSON-CSV-files/clean_csv_main_unclean.csv"
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
            # Read in cleaned dataset
            df = pd.read_csv(self.file_name)

            """
                Functions for making a vocabulary of all the words of the twitter data
                
                1. Use the Count Vectoriser to create a Document-Term matrix
                2. We specify to only include words that appear in less than 80% of and appear in at least 2 documents
            """

            # https://stackabuse.com/python-for-nlp-topic-modeling/
            tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2)
            doc_term_matrix = tf_vectorizer.fit_transform(df["text"].values.astype('U'))

            """
                Latent Dirichlet Allocation 
                
                1. Include the number of topics we want to create
                2. Create a Visualisation in pyLDAvis
            """

            # Create 20 Topics
            numTopics = 20

            LDA = LatentDirichletAllocation(n_components=20, learning_method='online', random_state=42,
                                            n_jobs=-1)
            LDA_output = LDA.fit(doc_term_matrix)

            visualisation = pyLDAvis.sklearn.prepare(LDA_output, doc_term_matrix, tf_vectorizer, mds='tsne')
            pyLDAvis.save_html(visualisation, 'LDA_Visualization_Main_Unclean.html')

            """
            After finding a topic of interest, sort the data based on this topic            
            """
            newIdx = translateLdaIdx(doc_term_matrix, visualisation)
            visDF = createDenseMat(doc_term_matrix, newIdx)

            """
            Create a Histogram of Topic 12
            """

            fig = plt.figure(figsize=(4, 4), dpi=1600)
            ax = plt.subplot(111)
            plt.hist(visDF.topic12, bins=20)

            ax.tick_params(labelsize=16)
            ax.set_xlabel('Gun Violence Topic Value', fontsize=24, fontweight='bold')
            ax.set_ylabel('Tweets (Topic 12)', fontsize=24, fontweight='bold')
            plt.show()
            x = fig.tight_layout()

            """ 
            Create Word-Art of Topic 12
            """

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    def k_means(self):
        try:
            # https://github.com/Ashwanikumarkashyap/k-means-clustering-tweets-from-scratch/blob/master/main.py
            # Allow user to find the CSV file they wish to apply K-Means to
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            df = pd.read_csv(self.file_name)
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(df)
            n_clusters = 10  # Create 10 clusters
            clf = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
            data = clf.fit(X)
            centroids = clf.cluster_centers_
            everything = concatenate((X.todense(), centroids))

            print(f"Top terms per cluster: ")
            order_centroids = data.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names()
            for i in range(n_clusters):
                print("Cluster %d:" % i),
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind])
                print()

            """
            # https://stackoverflow.com/questions/43541187/how-can-i-plot-a-kmeans-text-clustering-result-with-matplotlib/45510082
            tsne_init = 'pca'
            tsne_perplexity = 20.0
            tsne_early_exaggeration = 4.0
            tsne_learning_rate = 1000
            model = TSNE(n_components=2, random_state=random_state, init=tsne_init,
                         perplexity=tsne_perplexity,
                         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

            transformed_everything = model.fit_transform(everything)
            print(transformed_everything)
            plt.scatter(transformed_everything[:-n_clusters, 0], transformed_everything[:-n_clusters, 1], marker='x')
            plt.scatter(transformed_everything[-n_clusters:, 0], transformed_everything[-n_clusters:, 1], marker='o')

            plt.show()

            # https://pythonprogramminglanguage.com/kmeans-text-clustering/
            plt.show()
            """
        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)


# Driver Code
root = Tk()
root.title('Twitter Visualisation Tool')

obj = TwitterApplication(root)
root.geometry('600x300')
root.mainloop()
