import pandas as pd
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox as msg
import json
from pandastable import Table
# from tkintertable import TableCanvas


class TwitterApplication:

    def __init__(self, root):

        self.root = root
        self.filename = ''
        self.f = Frame(self.root, height=200, width=300)

        # Place the frame on root window
        self.f.pack()

        # Creating label widgets
        self.message_label = Label(self.f,
                                   text='Twitter Tool',
                                   font=('Arial', 19, 'bold'))

        # Buttons
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

        # Placing the widgets using grid manager
        self.message_label.grid(row=1, column=0)
        self.convert_button.grid(row=4, column=0,
                                 padx=0, pady=15)
        self.k_means_button.grid(row=4, column=1,
                                 padx=10, pady=15)

    def convert_csv_to_json(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a CSV file',
                                                        filetypes=(('csv file', '*.csv'),
                                                                   ('csv file', '*.csv')))

            col_list = ["text"]
            df = pd.read_csv(self.file_name, usecols=col_list)

            # Next - Pandas DF to Excel file on disk
            if len(df) == 0:
                msg.showinfo('No Rows Selected', 'CSV has no rows')
            else:
                # msg.showinfo('CSV file selected', 'CSV file selected')
                createjson = df.to_json()

                # print(createjson)
                with open("../CSV and JSON files/TwitterTextData.json", 'w') as write_file:
                    json.dump(createjson, write_file)
                msg.showinfo('JSON file created', 'JSON file created')

        except FileNotFoundError as e:
            msg.showerror('Error opening file', e)

    def k_means(self):
        try:
            self.file_name = filedialog.askopenfilename(initialdir='/Desktop',
                                                        title='Select a JSON file',
                                                        filetypes=(('json file', '*.json'),
                                                                   ('json file', '*.json')))

            df = pd.read_json(self.file_name)

            if len(df) == 0:
                msg.showinfo('No data', 'No data')
            else:
                msg.showinfo('Json file selected', 'Json file selected')

            """
            # Now display the DF in 'Table' object
            # under 'pandastable' module
            self.f2 = Frame(self.root, height=200, width=300)
            self.f2.pack(file=BOTH, expand=1)
            self.table = Table(self.f2, dataframe=df, read_only=True)
            self.table.show()
            """

        except FileNotFoundError as e:
            print(e)
            msg.showerror('Error opening file')


# Driver Code
root = Tk()
root.title('Tweet Application')

obj = TwitterApplication(root)
root.geometry('600x600')
root.mainloop()
