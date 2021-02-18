import pandas as pd
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg
import json, csv
import re
import matplotlib as plt
import numpy as np
import sklearn.cluster import KMeans

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

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
                                   text='Twitter Tool',
                                   font=('Arial', 19, 'bold'))
        self.message_label2 = Label(self.f,
                                    text='Clean .csv file.',
                                    font=('Arial', 12, 'underline'))
        self.message_label3 = Label(self.f,
                                    text='Convert .csv file to .json.',
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

        self.convert_button = Button(self.f,
                                     text='Convert',
                                     font=('Arial', 14),
                                     bg='Orange',
                                     fg='Black',
                                     command=self.convert_csv_to_json)

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
                                 command=self.k_means)

        # Placing the widgets using grid manager
        self.message_label.grid(row=0, column=1)
        self.message_label2.grid(row=2, column=1)
        self.clean_button.grid(row=4, column=1,
                               padx=0, pady=15)
        self.message_label3.grid(row=5, column=1)
        self.convert_button.grid(row=6, column=1,
                                 padx=0, pady=15)
        self.message_label4.grid(row=8, column=1)
        self.k_means_button.grid(row=9, column=0,
                                 padx=10, pady=15)
        self.lda_button.grid(row=9, column=2,
                             padx=10, pady=15)

    def lemmatize_text(self):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(self)]

    def clean_csv(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            col_list = ["text"]
            clean_tweets = []
            df = pd.read_csv(self.file_name, usecols=col_list)
            stop_words = set(stopwords.words('english'))

            if len(df) == 0:
                msg.showinfo('No Rows Selected', 'CSV is empty.')
            else:

                # msg.showinfo('File selected', 'CSV file selected to be cleaned.')
                for tweet in df["text"]:
                    """ Code taken to clean tweet data was taken from:
                    https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/analyze-tweet-sentiment-in-python/
                    Last accessed: 03/02/2021
                    
                    To remove URLs and non-alphanumeric characters
                    """
                    tweet = (re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(rt)", "", tweet.lower()).split())
                    # Removing Stop Words
                    clean_tweets.append(set(tweet).difference(stop_words))

                df['text'] = clean_tweets
                df.to_csv("../JSON-CSV-files/clean_csv.csv", index=False)
                msg.showinfo('File', 'File has been cleaned. Ready to be converted.')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

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

    def k_means(self):
        try:
            # https://github.com/Ashwanikumarkashyap/k-means-clustering-tweets-from-scratch/blob/master/main.py

            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a JSON file',
                                                        filetypes=(('json file', '*.json'),
                                                                   ('json file', '*.json')))

            df = pd.read_json(self.file_name)

            if len(df) == 0:
                msg.showinfo('No data', 'No data')
            else:
                msg.showinfo('Json file selected', 'Json file selected')

        except FileNotFoundError as e:
            print(e)
            msg.showerror('Error opening file')

    """    
    def lda(self):
    
    """


# Driver Code
root = Tk()
root.title('Twitter Visualisation Tool')

obj = TwitterApplication(root)
root.geometry('600x600')
root.mainloop()
