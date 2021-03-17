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
        self.message_label4.grid(row=8, column=1)
        self.k_means_button.grid(row=9, column=0,
                                 padx=10, pady=15)
        self.lda_button.grid(row=9, column=2,
                             padx=10, pady=15)

    def lemmatize_text(self):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in self.split()]

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
            newStopWords = ['amp', 'rt', 'http']
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
                    array.append(filtered_sentence)

                    # lemmatizer.lemmatize()
                    """
                    array = []
                    for word in filtered_sentence:
                        test = lemmatizer.lemmatize(word, wordnet.VERB)
                        stem = ps.stem(test)
                        nltk_tokens = nltk.word_tokenize(stem)
                    array.append(nltk_tokens) """

            # with open('../JSON-CSV-files/clean_csv.csv', 'w') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(array)

            # Create a new dataframe of the cleaned dataset
            df['text'] = array
            # Add to file called clean_csv
            clean_file = r"../JSON-CSV-files/clean_file_LDA.csv"
            df.to_csv(clean_file, index=False)
            msg.showinfo('File', 'File has been cleaned. Ready to be converted.')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

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

            df = pd.read_csv(self.file_name)
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(df)
            n_clusters = 10  # Create 10 clusters
            random_state = 1
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

            """
            # plt.subplots(figsize=(20,20))
            wordcloud = WordCloud(stopwords=stopwords,
                                  background_color='white',
                                  width=512,
                                  height=384
                                  ).generate(model)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')"""

            plt.show()

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

            df = pd.read_csv(self.file_name)

            # https://stackabuse.com/python-for-nlp-topic-modeling/
            # count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
            count_vect = CountVectorizer(max_df=0.8, min_df=2)
            doc_term_matrix = count_vect.fit_transform(df["text"].values.astype('U'))

            # Create 10 Topics
            number_of_topics = 10
            LDA = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', random_state=42,
                                            n_jobs=-1)
            LDA_output = LDA.fit(doc_term_matrix)
            # print(LDA_output)

            """
            for i, topic in enumerate(LDA.components_):
                print(f"Top 10 words for topic #{i}")
                print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
                print("\n")"""

            # Visualise the topics
            # pyLDAvis.enable_notebook()

            visualisation = pyLDAvis.sklearn.prepare(LDA_output, doc_term_matrix, count_vect, mds='tsne')
            # visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)


# Driver Code
root = Tk()
root.title('Twitter Visualisation Tool')

obj = TwitterApplication(root)
root.geometry('600x600')
root.mainloop()
