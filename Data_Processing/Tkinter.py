from tkinter import *
from tkinter import ttk
from tkinter.messagebox import *
from tkinter import messagebox
from PIL import ImageTk, Image
import Data_Processing.ReviewManager as RM
import Data_Processing.CleanDataSets as CD
import Data_Processing.MongoData as PMD
import Data_Processing.Flatten as FL
import Data_Processing.ReviewsSegmentation as RS
import Data_Processing.Clustering as CL
import os
import time
from datetime import date
import pandas as pd


class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        self.master.title("Web Scraping")
        self.pack(fill=BOTH, expand=1)
        self.prompt = Label(self, text="Choisir un modèle de voiture:")
        self.prompt.place(x=10, y=0)

        rm = RM.ReviewManagement()
        rm.populateReviews()
        products = sorted(list(rm.getCars()))

        def callbackFunc(event):
            global car_model
            car_model = self.productsCombo.get()

        # creating the Combobox and populating it with products
        self.productsCombo = ttk.Combobox(self, values=products, state=READABLE)
        self.productsCombo.place(x=200, y=0)
        self.productsCombo.bind("<<ComboboxSelected>>", callbackFunc)

        self.prompt = Label(self, text="Choisir un pays:")
        self.prompt.place(x=10, y=50)

        countries = sorted(list(rm.getCountries()))
        countries.insert(0, 'ALL')

        def chooseCountry(event):
            global country
            country = self.countriesCombo.get()

        # creating the Combobox and populating it with products
        self.countriesCombo = ttk.Combobox(self, values=countries, state=READABLE)
        self.countriesCombo.place(x=200, y=50)
        self.countriesCombo.bind("<<ComboboxSelected>>", chooseCountry)

        Label(self, text="Choisir le(s) Type(s) de Pages: ").place(x=10, y=100)
        page_types = StringVar()
        page_types.set("News Blog Forum General Review Microblog Image Facebook Twitter")
        lstbox = Listbox(self, listvariable=page_types, selectmode=MULTIPLE)
        lstbox.place(x=200, y=100)
        global page_type_list
        page_type_list = list()

        def select():
            selection = lstbox.curselection()
            for i in selection:
                entree = lstbox.get(i)
                page_type_list.append(entree)

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.place(x=200, y=265)

        motcle = StringVar()
        Label(self, text="Choisir un mot clé: ").place(x=10, y=300)
        Entry(self, textvariable=motcle, width=50).place(x=200, y=300)

        startdate = StringVar()
        Label(self, text="Choisir la date de début: ").place(x=10, y=350)
        Entry(self, textvariable=startdate, width=50).place(x=200, y=350)

        enddate = StringVar()
        today = date.today()
        Label(self, text="Choisir la date de fin: ").place(x=10, y=400)
        end_date_entry = Entry(self, textvariable=enddate, width=50)
        end_date_entry.insert(0, today.strftime("%Y-%m-%d"))
        end_date_entry.place(x=200, y=400)

        def search_data():
            if not startdate.get():
                messagebox.showerror("Error", "Veuillez choisir une date de début.")
            if not enddate.get():
                messagebox.showerror("Error", "Veuillez choisir une date de fin.")
            if not self.productsCombo.get():
                messagebox.showerror("Error", "Veuillez choisir un modèle de voiture.")
            if len(page_type_list) == 0:
                messagebox.showerror("Error", "Veuillez choisir le(s) type(s) de pages.")
            else:

                global message_dataset_contruction
                message_dataset_contruction = rm.getReviewsByCar(car_model, country, page_type_list, motcle.get(),
                                                                 startdate.get(), enddate.get())

                if message_dataset_contruction not in ["Erreur lors du processus de contruction de la base.",
                                                       "Aucun élement correspondant à vos critères dans la Base de données."
                                                       ]:
                    messagebox.showinfo("Construction de la Base de données", message_dataset_contruction)

                    def clean_dataset():
                        path = os.getcwd()
                        global meesage_data_cleaning
                        meesage_data_cleaning = CD.clean_dataset(path+"\Outputs\DataSets\Dataset_clean", path+
                                                                 "\Outputs\DataSets\Dataset.csv")

                        if meesage_data_cleaning != "Erreur lors du processus de Nettoyage de la base.":
                            DATA_PATH = path+r"\Outputs\DataSets\Dataset_clean.csv"
                            global result_flatten
                            result_flatten = FL.Reviews_Flatten(DATA_PATH, path)

                            if result_flatten != "error":
                                messagebox.showinfo("Transformation des données", "Terminé avec succès")

                                global minimum_term_frequency
                                minimum_term_frequency = DoubleVar()
                                minimum_term_frequency.set(0.005)
                                Label(self, text="Classification des données:").place(x=200, y=550)
                                Label(self, text="Minimum term frequency:").place(x=300, y=585)
                                minimum_term_frequency_entry = Entry(self, textvariable=minimum_term_frequency, width=5)
                                minimum_term_frequency_entry.place(x=465, y=585)

                                def set_minimum_term_frequency():
                                    Minimum_term_frequency = minimum_term_frequency.get()
                                    global meesage_keywords
                                    if Minimum_term_frequency != "":
                                        meesage_keywords = RS.Cluster_key_words(
                                            path+r"\Outputs\DataSets\Reviews_final.csv", path)
                                        global message_plot_keywords
                                        message_plot_keywords = RS.Plot_Clusters_trained(
                                            path+r"\Outputs\DataSets\Data_clustered.csv",
                                            path+r"\Outputs\Classes\.",
                                            Minimum_term_frequency)
                                        RS.Plot_Time_Series(path+r"\Outputs\DataSets\Data_clustered.csv", path)
                                        RS.wordcloud_cluster_byIds(path+r"\Outputs\DataSets\Data_clustered.csv", path)

                                        # create a data frame for each group (data + doc2vec)
                                        clusters = pd.read_csv(path+r"\Outputs\DataSets\Data_clustered.csv",
                                                               error_bad_lines=False,
                                                               encoding='utf-8')
                                        clusters['Cluster'] = clusters['Cluster'].astype(str)
                                        categories = list(set(clusters['Cluster']))
                                        try:
                                            categories.remove("nan")
                                        except:
                                            pass

                                        for cat in categories:
                                            try:
                                                directory, file_name, outputfile_name = CL.main(path, cat)
                                                newwin = Toplevel(root)
                                                img = ImageTk.PhotoImage(
                                                    Image.open(directory+r"\\ElbowCurve.png").resize((650, 300),
                                                                                                     Image.ANTIALIAS))
                                                panel = Label(newwin, image=img)
                                                panel.pack(side="bottom", fill="both", expand="yes")

                                                def clustering2():
                                                    global nbr_clusters
                                                    nbr_clusters = number_clusters.get()

                                                    CL.clustering(outputfile_name, file_name,
                                                                  directory+r"\text2kw_top_n_keywords.csv",
                                                                  nbr_clusters)
                                                    CL.Plot_Clusters_Kmeans(outputfile_name, nbr_clusters,
                                                                            directory+r"\\")
                                                    CL.wordcloud_cluster_byIds_Kmeans(directory+r"\\", outputfile_name,
                                                                                      nbr_clusters)
                                                    newwin.destroy()

                                                number_clusters = IntVar()
                                                Label(newwin,
                                                      text="Choisir le Nombre de clusters au sein de la classe "+
                                                           str(cat)+" : ").place(x=0, y=0)
                                                submit_number_clusters = Button(newwin, text='Entrée',
                                                                                command=clustering2)
                                                Entry(newwin, textvariable=number_clusters, width=10).place(x=500, y=0)
                                                submit_number_clusters.bind('<Return>', number_clusters)
                                                submit_number_clusters.place(x=500, y=25)
                                                # size of the window
                                                newwin.geometry("700x400")
                                                root.wait_window(newwin)
                                            except:
                                                pass

                                submit_minimum_term_frequency = Button(self, text='Entrée',
                                                                       command=set_minimum_term_frequency)
                                submit_minimum_term_frequency.bind('<Return>', set_minimum_term_frequency)
                                submit_minimum_term_frequency.place(x=465, y=550)

                            else:
                                messagebox.showinfo("Transformation des données", "Terminé avec succès")

                        else:
                            messagebox.showerror("Error", meesage_data_cleaning)

                    Label(self, text="Nettoyage de la base de données: ").place(x=200, y=500)
                    submit_clean_dataset = Button(self, text='Entrée', command=clean_dataset)
                    submit_clean_dataset.bind('<Return>', clean_dataset)
                    submit_clean_dataset.place(x=465, y=500)
                else:
                    messagebox.showerror("Error", message_dataset_contruction)

        Label(self, text="Construction de la base de données: ").place(x=200, y=450)
        submit_project_name = Button(self, text='Entrée', command=search_data)
        submit_project_name.bind('<Return>', search_data)
        submit_project_name.place(x=465, y=450)


root = Tk()

# size of the window
root.geometry("700x650")

app = Window(root)
root.mainloop()
