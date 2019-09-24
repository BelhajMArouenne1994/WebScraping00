from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from tkinter.ttk import *
from Collecting_Data.BrandWatch.brandy import *
from tkinter import messagebox
from functools import partial
from pandastable import Table, TableModel, VerticalScrolledFrame
from tkintertable import TableCanvas, TableModel
import pandas as pd
import time
import os
import threading


class Scraping(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)
        container.master.title("Web Scraping")

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (
                StartPage, BrandWatch, FaceBook, Twitter, AutomobilePropre, ElbilForum, GoingElectric, SpeakEV,
                Caradisiac):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Page d'acceuil")
        label.pack(pady=10, padx=10)

        button1 = Button(self, text="Scraping avec BrandWatch", command=lambda: controller.show_frame(BrandWatch))
        button1.pack(fill=X)

        button2 = Button(self, text="Scraping des groupes Facebook", command=lambda: controller.show_frame(FaceBook))
        button2.pack(fill=X)

        button3 = Button(self, text="Scraping de Twitter", command=lambda: controller.show_frame(Twitter))
        button3.pack(fill=X)

        button4 = Button(self, text="Scraping du Forum Automobile Propre",
                         command=lambda: controller.show_frame(AutomobilePropre))
        button4.pack(fill=X)

        button5 = Button(self, text="Scraping du Forum ElbilForum", command=lambda: controller.show_frame(ElbilForum))
        button5.pack(fill=X)

        button6 = Button(self, text="Scraping du Forum GoingElectric",
                         command=lambda: controller.show_frame(GoingElectric))
        button6.pack(fill=X)

        button7 = Button(self, text="Scraping du Forum SpeakEV", command=lambda: controller.show_frame(SpeakEV))
        button7.pack(fill=X)

        button8 = Button(self, text="Scraping du Forum Caradisiac", command=lambda: controller.show_frame(Caradisiac))
        button8.pack(fill=X)


class BrandWatch(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="BrandWatch")

        def show():
            user_name = username.get()
            pass_word = password.get()
            access_token = get_new_key(str(user_name), str(pass_word))
            label_token = Label(self, text="Access Token: "+access_token)
            label_token.grid(row=3, column=1)

            if access_token != 'username-password incorrect(s)':

                def selectProject(event):

                    iid = tree_projects.identify_row(event.y)
                    if iid:
                        global project_name_s
                        project_name_s = tree_projects.item(iid)['values'][0]
                        project_box = Entry(self, textvariable=project_name_s, width=100)
                        project_box.delete(0, END)
                        project_box.insert(0, project_name_s)
                        project_box.grid(row=7, column=1)
                    else:
                        pass

                extra_window = Toplevel(self)
                extra_window.geometry("+300+300")

                extra_window.title("Liste des projets:")
                Label(self, text="Projects list")
                df = boot_brandy(access_token)
                df_col = df.columns.values
                tree_projects = ttk.Treeview(extra_window)
                tree_projects['show'] = 'headings'
                tree_projects["columns"] = df_col
                counter = len(df)

                # generating for loop to create columns and give heading to them through df_col var.
                for x in range(len(df_col)):
                    tree_projects.column(x, width=100)
                    tree_projects.heading(x, text=df_col[x])

                    # generating for loop to print values of dataframe in treeview column.
                    for i in range(counter):
                        tree_projects.insert('', i, values=df.iloc[i, :].tolist())

                global project_name_s
                project_name_s = tree_projects.bind("<ButtonRelease-1>", selectProject)

                tree_projects.pack()

            else:
                showwarning('Résultat', 'Mot de passe incorrect.\nVeuillez recommencer !')

            def show_project_list():
                global empty_project
                try:
                    def selectItem(event):
                        iid = tree.identify_row(event.y)
                        try:
                            if iid:
                                global query_id_s
                                query_id_s = tree.item(iid)['values'][2]
                                query_id_box = Entry(self, textvariable=queryid, width=100)
                                query_id_box.delete(0, END)
                                query_id_box.insert(0, query_id_s)
                                query_id_box.grid(row=13, column=1)
                                extra_window.destroy()
                        except:
                            pass

                    try:
                        project_id = get_project_id_from_name(df, project_name_s)
                        query_list = get_query_id(project_id, access_token)
                        query_list_col = query_list.columns.values
                        extra_window_queries = Toplevel(self)
                        extra_window_queries.geometry("+300+300")
                        extra_window_queries.title("Liste des queries:")

                        tree = ttk.Treeview(extra_window_queries)
                        tree['show'] = 'headings'
                        tree["columns"] = query_list_col

                        # generating for loop to create columns and give heading to them through df_col var.
                        counter = len(query_list)
                        x = 0
                        tree.column(x, width=100)
                        tree.heading(x, text=query_list_col[x])

                        # generating for loop to print values of dataframe in treeview column.
                        for i in range(counter):
                            tree.insert('', i, values=query_list.iloc[i, :].tolist())
                        tree.bind("<ButtonRelease-1>", selectItem)
                        tree.pack()

                        empty_project = False

                    except:
                        empty_project = True
                        showwarning('Warning', 'Ce projet est vide.\nVeuillez recommencer !')

                    def Streaming_Data():
                        extra_window_queries.destroy()

                        project_id = get_project_id_from_name(df, project_name_s)
                        start_date = startdate.get()
                        end_date = enddate.get()
                        model = car_model.get()
                        query_ID = queryid.get()

                        def foo():
                            global streamed_data_info
                            streamed_data_info = stream_data(model, start_date, end_date, project_id, query_ID,
                                                             access_token, page_type_list)  # simulate some work

                        def start_foo_thread(event):
                            global foo_thread
                            foo_thread = threading.Thread(target=foo)
                            foo_thread.daemon = True
                            progressbar.start()
                            foo_thread.start()
                            self.after(20, check_foo_thread)

                        def check_foo_thread():
                            if foo_thread.is_alive():
                                self.after(20, check_foo_thread)
                            else:
                                progressbar.stop()
                                messagebox.showinfo("Scraping terminé", streamed_data_info)
                                extra_window_streaming.destroy()

                        extra_window_streaming = Toplevel()
                        extra_window_streaming.geometry("+300+300")
                        extra_window_streaming.title("Avancement du scraping")
                        Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                        progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                        progressbar.grid()

                        ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                                   command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                    if counter > 0 and empty_project == False:

                        Label(self, text="Choisir query ID: ", width=30).grid(row=13)
                        query_id_box = Entry(self, textvariable=queryid, width=100)
                        query_id_box.grid(row=13, column=1)

                        Label(self, text="Choisir la date de début: ", width=30).grid(row=14)
                        Entry(self, textvariable=startdate, width=100).grid(row=14, column=1)

                        Label(self, text="Choisir la date de fin: ", width=30).grid(row=15)
                        Entry(self, textvariable=enddate, width=100).grid(row=15, column=1)

                        Label(self, text="Modèle: ", width=30).grid(row=16)
                        Entry(self, textvariable=car_model, width=100).grid(row=16, column=1)

                        Label(self, text="Choisir le(s) Type(s) de Pages: ", width=30).grid(row=17, sticky=N)

                        page_types = StringVar()
                        page_types.set("News Blog Forum General Review Microblog Image")

                        lstbox = Listbox(self, listvariable=page_types, selectmode=MULTIPLE)
                        lstbox.grid(row=17, column=1, sticky='SW')

                        def select():
                            global page_type_list
                            page_type_list = list()
                            selection = lstbox.curselection()
                            for i in selection:
                                entree = lstbox.get(i)
                                page_type_list.append(entree)

                            submit_project = Button(self, text='Entrée', command=Streaming_Data)
                            submit_project.bind('<Return>', Streaming_Data)
                            submit_project.grid(row=19, column=2)

                        btn = ttk.Button(self, text="Sélectionner", command=select)
                        btn.grid(row=17, column=1, sticky=SW)



                except:
                    if not empty_project:
                        showwarning('Warning', 'Projet introuvable.\nVeuillez recommencer !')

            if access_token != 'username-password incorrect(s)':
                Label(self, text="Choisir votre projet: ", width=30).grid(row=7)

                bool_project = False
                while not bool_project:
                    project_box = Entry(self, textvariable=project_name_s, width=100)
                    try:
                        submit_project_name = Button(self, text='Entrée', command=show_project_list)
                        bool_project = True
                    except:
                        showwarning('Warning', 'Projet introuvable.\nVeuillez recommencer !')

                if submit_project_name != "":
                    username_box.delete(0, END)
                    username_box.insert(0, 'marouenne.belhaj@renault.com')
                project_box.grid(row=7, column=1)
                submit_project_name.bind('<Return>', show_project_list)
                submit_project_name.grid(row=8, column=2, sticky='NESW')

        username = StringVar()  # Password variable
        password = StringVar()  # Password variable
        project_name_s = StringVar()  # Project_name variable
        queryid = StringVar()
        startdate = StringVar()
        enddate = StringVar()
        car_model = StringVar()

        Label(self, text='User Name:', width=30).grid(row=0)
        Label(self, text='Password :', width=30).grid(row=1)

        username_box = Entry(self, textvariable=username, width=100)
        username_box.insert(0, 'marouenne.belhaj@renault.com')
        username_box.grid(row=0, column=1)

        Entry(self, textvariable=password, show="*", width=100).grid(row=1, column=1)

        button = Button(self, text="Start Page", command=lambda: controller.show_frame(StartPage))
        button.grid(row=20, column=2)

        submit = Button(self, text='Entrée', command=show)
        submit.bind('<Return>', show)
        submit.grid(row=3, column=2, sticky='NESW')


class FaceBook(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping des groupes Facebook")

        def Streaming_Data():

            def Facebook():
                User_name = username.get()
                Password = password.get()
                Group_URL = groupURL.get()
                Lang = lang_choosen
                Country = country_choosen
                Car = car.get()
                cmd = 'cd Facebook  && cd Facebook && cd && ' \
                      'scrapy crawl cars -a email="{0}" -a password="{1}" -a url="{2}" -a language="{3}" -a ' \
                      'country="{4}" -a car="{5}"' \
                    .format(User_name, Password, Group_URL, Lang, Country, Car)
                os.system(cmd)

            def start_foo_thread(event):
                global foo_thread
                foo_thread = threading.Thread(target=Facebook)
                foo_thread.daemon = True
                progressbar.start()
                foo_thread.start()
                self.after(20, check_foo_thread)

            def check_foo_thread():
                if foo_thread.is_alive():
                    self.after(20, check_foo_thread)
                else:
                    progressbar.stop()
                    messagebox.showinfo("Info", "Scraping terminé.")
                    extra_window_streaming.destroy()

            extra_window_streaming = Toplevel()
            extra_window_streaming.geometry("+300+300")
            extra_window_streaming.title("Avancement du scraping")
            Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
            progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
            progressbar.grid()

            ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                       command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

        username = StringVar()
        password = StringVar()
        groupURL = StringVar()
        lang = StringVar()
        car = StringVar()

        Label(self, text="1 - Cette fonctionnalité sert à scraper les groupes Facebook. \n"
                         "2 - Avant de commencer, assurez vous de rejoindre le groupe en question. \n"
                         "3 - L'URL du groupe doit être sous cette forme: https://mbasic.facebook.com/groups/"
                         "...\n", font='Helvetica 10 bold').grid(row=0, column=1)

        Label(self, text='User Name:', width=30).grid(row=1)
        Label(self, text='Password :', width=30).grid(row=2)
        Label(self, text='URL du groupe Facebook :', width=30).grid(row=3)
        Label(self, text='Langue:', width=30).grid(row=4, sticky='NW')
        Label(self, text='Pays:', width=30).grid(row=4, column=1, sticky='NE')
        Label(self, text='Modèle:', width=30).grid(row=6)

        Entry(self, textvariable=username, width=100).grid(row=1, column=1)
        Entry(self, textvariable=password, show="*", width=100).grid(row=2, column=1)
        Groupe_URL_box = Entry(self, textvariable=groupURL, width=100)
        Groupe_URL_box.insert(0, 'https://mbasic.facebook.com/groups/...')
        Groupe_URL_box.grid(row=3, column=1)

        lang = StringVar()
        lang.set("Anglais Francais Allemand Italien Norvégien Russe Espagnol Turc Japonais")

        scrollbar_lang = Scrollbar(self, orient=VERTICAL)
        lstbox_lang = Listbox(self, listvariable=lang, selectmode=SINGLE, yscrollcommand=scrollbar_lang.set)
        scrollbar_lang.config(command=lstbox_lang.yview)
        lstbox_lang.grid(row=4, column=1, sticky='NSW')
        scrollbar_lang.grid(row=4, column=1, sticky='NSW', padx=125)

        def select():
            global lang_choosen
            selection = lstbox_lang.curselection()
            languages = ["en", "fr", "de", "it", "no", "ru", "es", "tr", "ja"]
            for i in selection:
                lang_choosen = languages[i]

        btn_lang = ttk.Button(self, text="Sélectionner", command=select)
        btn_lang.grid(row=5, column=1, sticky="SW")

        countries = StringVar()
        countries.set("United_States France Germany United_Kingdom Spain Italy Netherlands "
                      "Ireland Brazil Switzerland Australia Norway Sweden Belgium "
                      "Austria Romania Portugal Russia Turkey Canada Denmark Hungary "
                      "New_Zealand Finland Mexico China Japan")
        scrollbar_country = Scrollbar(self, orient=VERTICAL)
        lstbox_country = Listbox(self, listvariable=countries, selectmode=SINGLE, yscrollcommand=scrollbar_country.set)
        scrollbar_country.config(command=lstbox_country.yview)
        scrollbar_country.grid(row=4, column=2, sticky="NSW")
        lstbox_country.grid(row=4, column=1, sticky="E")

        def select_country():
            pays = ["United States", "France", "Germany", "United Kingdom", "Spain", "Italy", "Netherlands",
                    "Republic of Ireland", "Brazil", "Switzerland", "Australia", "Norway", "Sweden", "Belgium",
                    "Austria", "Romania", "Portugal", "Russia", "Turkey", "Canada", "Denmark", "Hungary",
                    "New Zealand", "Finland", "Mexico", "China", "Japan"]
            global country_choosen
            selection = lstbox_country.curselection()
            for i in selection:
                country_choosen = pays[i]

        btn_country = ttk.Button(self, text="Sélectionner", command=select_country)
        btn_country.grid(row=5, column=1, sticky="E")

        Entry(self, textvariable=car, width=100).grid(row=6, column=1, sticky='SW')

        submit = Button(self, text='Entrée', command=Streaming_Data)
        submit.bind('<Return>', Streaming_Data)
        submit.grid(row=7, column=1, sticky='NESW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=8, column=1, sticky='NESW')


class Twitter(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping de Twitter")

        def Streaming_Data():

            def Facebook():
                Query = query.get()
                Query = Query.replace(" ", "+")
                Lang = lang_choosen
                Car = car.get()
                cmd = 'cd TweetScraper  && cd TweetScraper && cd && ' \
                      'scrapy crawl TweetScraper -a query="{0}" -a lang="{1}" -a car="{2}"' \
                    .format(Query, Lang, Car)
                os.system(cmd)

            def start_foo_thread(event):
                global foo_thread
                foo_thread = threading.Thread(target=Facebook)
                foo_thread.daemon = True
                progressbar.start()
                foo_thread.start()
                self.after(20, check_foo_thread)

            def check_foo_thread():
                if foo_thread.is_alive():
                    self.after(20, check_foo_thread)
                else:
                    progressbar.stop()
                    messagebox.showinfo("Info", "Scraping terminé.")
                    extra_window_streaming.destroy()

            extra_window_streaming = Toplevel()
            extra_window_streaming.geometry("+300+300")
            extra_window_streaming.title("Avancement du scraping")
            Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
            progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
            progressbar.grid()

            ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                       command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

        query = StringVar()
        car = StringVar()

        Label(self, text='Query:', width=30).grid(row=1)
        Label(self, text='Langue:', width=30).grid(row=2, sticky='NW')
        Label(self, text='Modèle:', width=30).grid(row=4)

        Entry(self, textvariable=query, width=100).grid(row=1, column=1)
        Entry(self, textvariable=car, width=100).grid(row=4, column=1)

        lang = StringVar()
        lang.set("Anglais Francais Allemand Italien Norvégien Russe Espagnol Turc Japonais")

        scrollbar_lang = Scrollbar(self, orient=VERTICAL)
        lstbox_lang = Listbox(self, listvariable=lang, selectmode=SINGLE, yscrollcommand=scrollbar_lang.set)
        scrollbar_lang.config(command=lstbox_lang.yview)
        lstbox_lang.grid(row=2, column=1, sticky='NSW')
        scrollbar_lang.grid(row=2, column=1, sticky='NSW', padx=125)

        def select():
            global lang_choosen
            selection = lstbox_lang.curselection()
            languages = ["en", "fr", "de", "it", "no", "ru", "es", "tr", "ja"]
            for i in selection:
                lang_choosen = languages[i]

        btn_lang = ttk.Button(self, text="Sélectionner", command=select)
        btn_lang.grid(row=3, column=1, sticky="SW")

        submit = Button(self, text='Entrée', command=Streaming_Data)
        submit.bind('<Return>', Streaming_Data)
        submit.grid(row=5, column=1, sticky='NESW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=6, column=1, sticky='NESW')


class AutomobilePropre(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping du Forum Automobile Propre")

        cars = {'Audi': {'Audi_e-tron': 'https://forums.automobile-propre.com/forums/audi-e-tron-75/',
                         'Audi_A3_e-tron': 'https://forums.automobile-propre.com/forums/audi-a3-e-tron-45/'},
                'BMW': {'BMW_i3': 'https://forums.automobile-propre.com/forums/bmw-i3-28/',
                        'BMW_i8': 'https://forums.automobile-propre.com/forums/bmw-i8-43/',
                        'Mini_Cooper_SE': 'https://forums.automobile-propre.com/forums/mini-cooper-se-109/'},
                'Citroen': {
                    'Citroen_E-Mehari': 'https://forums.automobile-propre.com/forums/citro%C3%ABn-em%C3%A9hari-81/',
                    'Citroen_DS_3_Crossback': 'https://forums.automobile-propre.com/forums/ds-3-crossback-e-tense-91/',
                    'Citroen_DS_7_Crossback': 'https://forums.automobile-propre.com/forums/ds-7-crossback-e-tense-103/'},
                'Ford': {
                    'Ford_Focus_électrique': 'https://forums.automobile-propre.com/forums/ford-focus-%C3%A9lectrique-29/'},
                'Honda': {'Honda_e': 'https://forums.automobile-propre.com/forums/honda-e-105/'},
                'Hyundai': {
                    'Hyundai_Ioniq_électrique': 'https://forums.automobile-propre.com/forums/hyundai-ioniq-electrique-55/',
                    'Hyundai_Kona_électrique': 'https://forums.automobile-propre.com/forums/hyundai-kona-%C3%A9lectrique-71/',
                    'Hyundai_Ioniq_Hybride_rechargeable': 'https://forums.automobile-propre.com/forums/hyundai-ioniq-hybride-rechargeable-58/',
                    'Hyundai_Ioniq_hybride': 'https://forums.automobile-propre.com/forums/hyundai-ioniq-hybride-63/'},
                'Jaguar': {'Jaguar_i-Pace': 'https://forums.automobile-propre.com/forums/jaguar-i-pace-72/'},
                'KIA': {'Kia_Niro_EV': 'https://forums.automobile-propre.com/forums/kia-e-niro-%C3%A9lectrique-73/',
                        'Kia_Soul_EV': 'https://forums.automobile-propre.com/forums/kia-soul-ev-46/',
                        'KIA_Niro_hybride_rechargeable': 'https://forums.automobile-propre.com/forums/kia-niro-hybride-rechargeable-64/',
                        'KIA_Optima_hybride_rechargeable': 'https://forums.automobile-propre.com/forums/kia-optima-hybride-rechargeable-65/',
                        'KIA_Optima_hybride': 'https://forums.automobile-propre.com/forums/kia-optima-hybride-61/'},
                'Mercedes': {'Mercedes_EQC': 'https://forums.automobile-propre.com/forums/mercedes-eqc-76/',
                             'Mercedes_Class_A_Hydride_Rechargeable': 'https://forums.automobilepropre.com/forums/mercedes-class-a-hydride-rechargeable-66/'},
                'Mitsubishi': {
                    'Mitsubishi_Outlander_PHEV': 'https://forums.automobile-propre.com/forums/mitsubishi-outlander-phev-54/'},
                'Nissan': {'Nissan_e-NV200': 'https://forums.automobile-propre.com/forums/nissan-e-nv200-21/',
                           'Nissan_LEAF': 'https://forums.automobile-propre.com/forums/nissan-leaf-9/'},
                'Opel': {'Opel_Corsa-e': 'https://forums.automobile-propre.com/forums/opel-corsa-e-108/',
                         'Opel_Ampera-e': 'https://forums.automobile-propre.com/forums/opel-ampera-e-chevrolet-bolt-56/'},
                'Renault': {'Renault_Twizy': 'https://forums.automobile-propre.com/forums/renault-twizy-13/',
                            'Renault_Fluence_ZE': 'https://forums.automobile-propre.com/forums/renault-fluence-ze-10/',
                            'Renault_Zoe': 'https://forums.automobile-propre.com/forums/renault-zoe-14/',
                            'Renault_Kangoo_ZE': 'https://forums.automobile-propre.com/forums/renault-kangoo-ze-12/'},
                'Peugeot': {
                    'Peugeot_e-208_électrique': 'https://forums.automobile-propre.com/forums/peugeot-e-208-%C3%A9lectrique-90/',
                    'Peugeot_2008_électrique': 'https://forums.automobile-propre.com/forums/peugeot-2008-%C3%A9lectrique-101/',
                    'Peugeot_3008_hybride_rechargeable ': 'https://forums.automobile-propre.com/forums/peugeot-3008-hybride-rechargeable-92/'},
                'Porshe': {'Porsche_Taycan': 'https://forums.automobile-propre.com/forums/porsche-taycan-77/'},
                'Smart': {
                    'Smart_Fortwo_Electrique': 'https://forums.automobile-propre.com/forums/smart-fortwo-electrique-18/'},
                'Tesla': {'Tesla': 'https://forums.automobile-propre.com/forums/tesla-98/',
                          'Tesla_Roadster': 'https://forums.automobile-propre.com/forums/tesla-roadster-15/',
                          'Tesla_Model_S': 'https://forums.automobile-propre.com/forums/tesla-model-s-19/',
                          'Tesla_Model_X': 'https://forums.automobile-propre.com/forums/tesla-model-x-50/',
                          'Tesla_Model_3': 'https://forums.automobile-propre.com/forums/tesla-model-3-49/',
                          'Tesla_Model_Y': 'https://forums.automobile-propre.com/forums/tesla-model-y-70/',
                          'Tesla_Semi': 'https://forums.automobile-propre.com/forums/tesla-semi-69/',
                          'Tesla_Pick-up': 'https://forums.automobile-propre.com/forums/tesla-pick-up-74/'},
                'Volkswagen': {'Volkswagen_e-Golf': 'https://forums.automobile-propre.com/forums/volkswagen-e-golf-48/',
                               'Volkswagen_e-UP': 'https://forums.automobile-propre.com/forums/volkswagen-e-up-27/',
                               'Volkswagen_I.D.3': 'https://forums.automobile-propre.com/forums/volkswagen-id-3-96/',
                               'Volkswagen_Jetta_hybride': 'https://forums.automobile-propre.com/forums/volkswagen-jetta-hybride-26/',
                               'Volkswagen_Golf_GTE': 'https://forums.automobile-propre.com/forums/volkswagen-golf-gte-52/',
                               'Volkswagen_Passat_GTE': 'https://forums.automobile-propre.com/forums/volkswagen-passat-gte-68/'},
                'Volvo': {
                    'Volvo_XC40_hybride_rechargeable': 'https://forums.automobile-propre.com/forums/volvo-xc40-hybride-rechargeable-93/'},
                'Toyota': {'Toyota_C-HR': 'https://forums.automobile-propre.com/forums/toyota-c-hr-57/',
                           'Toyota_Corolla_hybride': 'https://forums.automobile-propre.com/forums/toyota-corolla-hybride-95/',
                           'Toyota_Prius_hybride': 'https://forums.automobile-propre.com/forums/toyota-prius-hybride-16/',
                           'Toyota_Yaris_hybride': 'https://forums.automobile-propre.com/forums/toyota-yaris-hybride-62/',
                           'Toyota_Prius_hybride_rechargeable': 'https://forums.automobile-propre.com/forums/toyota-prius-hybride-rechargeable-67/'}}

        constructeur = StringVar()
        modele = StringVar()

        Label(self, text="Constructeur automobile:", width=30).grid(row=1, sticky=N)

        constructeurs = StringVar()
        constructeurs.set(' '.join(list(cars.keys())))

        scrollbar_constructeurs = Scrollbar(self, orient=VERTICAL)
        lstbox_constructeurs = Listbox(self, listvariable=constructeurs, selectmode=SINGLE,
                                       yscrollcommand=scrollbar_constructeurs.set)
        scrollbar_constructeurs.config(command=lstbox_constructeurs.yview)
        lstbox_constructeurs.grid(row=1, column=1, sticky='NSW')
        scrollbar_constructeurs.grid(row=1, column=1, sticky='NSW', padx=125)

        def select():
            global constructeur
            selection = lstbox_constructeurs.curselection()
            constructeur = lstbox_constructeurs.get(selection)

            Label(self, text="Modèle:", width=30).grid(row=1, sticky=N)
            modeles = StringVar()
            modeles.set(' '.join(list(cars[constructeur].keys())))
            lstbox2 = Listbox(self, listvariable=modeles, selectmode=SINGLE)
            lstbox2.grid(row=1, column=2, sticky='SW')

            def select2():
                global modele
                global link
                selection2 = lstbox2.curselection()
                modele = lstbox2.get(selection2)
                link = cars[constructeur][modele]

                def streaming(link, modele):

                    def Elbilforum():
                        cmd = 'cd automobile_propre && cd automobile_propre && cd && ' \
                              'echo {0} && ' \
                              'scrapy crawl cars -a link="{0}" -a car="{1}"' \
                            .format(link, modele)
                        os.system(cmd)

                    def start_foo_thread(event):
                        global foo_thread
                        foo_thread = threading.Thread(target=Elbilforum)
                        foo_thread.daemon = True
                        progressbar.start()
                        foo_thread.start()
                        self.after(20, check_foo_thread)

                    def check_foo_thread():
                        if foo_thread.is_alive():
                            self.after(20, check_foo_thread)
                        else:
                            progressbar.stop()
                            messagebox.showinfo("Info", "Scraping terminé.")
                            extra_window_streaming.destroy()

                    extra_window_streaming = Toplevel()
                    extra_window_streaming.geometry("+300+300")
                    extra_window_streaming.title("Avancement du scraping")
                    Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                    progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                    progressbar.grid()

                    ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                               command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                if link != "" and modele != "":
                    streaming(link, modele)

            btn = ttk.Button(self, text="Sélectionner", command=select2)
            btn.grid(row=2, column=2, sticky=SW)

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.grid(row=2, column=1, sticky=SW)

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=3, column=2, sticky='NESW')


class ElbilForum(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping de ElbilForum")

        cars = {"Audi": {'Audi_e-tron': 'https://elbilforum.no/index.php?board=92.0',
                         'e-tron_Sportback': 'https://elbilforum.no/index.php?board=136.0'},
                "BMW": {'BMW_i3': 'https://elbilforum.no/index.php?board=66.0',
                        'BMW_iX3': 'https://elbilforum.no/index.php?board=131.0',
                        'Mini_Cooper_E': 'https://elbilforum.no/index.php?board=112.0'},
                "Citroen": {'Peugeot_e-208': 'https://elbilforum.no/index.php?board=135.0',
                            'Citroen_DS_3_Crossback': 'https://elbilforum.no/index.php?board=132.0',
                            'Opel_Ampera-e/Chevy_Bolt': 'https://elbilforum.no/index.php?board=87.0',
                            'Partner/Tepee/Berlingo': 'https://elbilforum.no/index.php?board=96.0',
                            'Opel_Corsa_e': 'https://elbilforum.no/index.php?board=142.0'},
                "Peugeot": {'Peugeot_e-208_électrique': 'https://elbilforum.no/index.php?board=135.0'},
                "PSA": {'Partner/Tepee/Berlingo': 'https://elbilforum.no/index.php?board=96.0'},
                "Hyundai": {'Hyundai_Ioniq_électrique': 'https://elbilforum.no/index.php?board=86.0',
                            'Hyundai_Kona_électrique': 'https://elbilforum.no/index.php?board=102.0'},
                "KIA": {'Kia_Niro_EV': 'https://elbilforum.no/index.php?board=115.0',
                        'Kia_Soul_EV': 'https://elbilforum.no/index.php?board=72.0'},
                "Plug-in_hybrid": {'Golf_GTE': 'https://elbilforum.no/index.php?board=85.0',
                                   'Mitsubishi_Outl._PH': 'https://elbilforum.no/index.php?board=74.0',
                                   'Toyota/Lexus_PH': 'https://elbilforum.no/index.php?board=73.0',
                                   'Volvos_PH': 'https://elbilforum.no/index.php?board=98.0',
                                   'BMWs_hybrider': 'https://elbilforum.no/index.php?board=100.0',
                                   'Opel_Ampera_PH': 'https://elbilforum.no/index.php?board=105.0',
                                   'Ioniq_PH': 'https://elbilforum.no/index.php?board=113.0'},
                "Mercedes": {'Mercedes_Class_B': 'https://elbilforum.no/index.php?board=84.0',
                             'Mercedes_EQC': 'https://elbilforum.no/index.php?board=110.0'},
                "Nissan": {'Nissan_Leaf_2018': 'https://elbilforum.no/index.php?board=106.0',
                           'Nissan_Leaf_2010-2017': 'https://elbilforum.no/index.php?board=46.0',
                           'Nissan_e-NV200': 'https://elbilforum.no/index.php?board=62.0'},
                "Opel": {'Opel_Ampera-e': 'https://elbilforum.no/index.php?board=87.0',
                         'Opel_Corsa_e': 'https://elbilforum.no/index.php?board=142.0'},
                "Renault": {'Renault_Zoe': 'https://elbilforum.no/index.php?board=47.0',
                            'Renault_Kangoo_ZE': 'https://elbilforum.no/index.php?board=81.0',
                            'Renault_Twizy': 'https://elbilforum.no/index.php?board=83.0'},
                "Volkswagen": {'Volkswagen_e-Golf': 'https://elbilforum.no/index.php?board=68.0',
                               'Volkswagen_e-up!': 'https://elbilforum.no/index.php?board=69.0',
                               'Volkswagen_I.D.Crozz': 'https://elbilforum.no/index.php?board=117.0',
                               'Volkswagen_I.D.Buzz': 'https://elbilforum.no/index.php?board=118.0',
                               'Volkswagen_I.D.Neo': 'https://elbilforum.no/index.php?board=126.0'},
                "Tesla": {'Tesla_Model_S': 'https://elbilforum.no/index.php?board=49.0',
                          'Tesla_Model_X': 'https://elbilforum.no/index.php?board=58.0',
                          'Tesla_Model_3': 'https://elbilforum.no/index.php?board=71.0',
                          'Tesla_Model_Y': 'https://elbilforum.no/index.php?board=121.0',
                          'Tesla_Roadster': 'https://elbilforum.no/index.php?board=56.0',
                          'Tesla_Pickup': 'https://elbilforum.no/index.php?board=122.0',
                          'Tesla_SW_og_AP': 'https://elbilforum.no/index.php?board=123.0',
                          'Tesla_Owners_Club': 'https://elbilforum.no/index.php?board=65.0'},
                "Skoda": {'Skoda_Citigo_iV': 'https://elbilforum.no/index.php?board=140.0',
                          'Skoda_Vision_iV': 'https://elbilforum.no/index.php?board=141.0'},
                "Porsche": {'Mission_E': 'https://elbilforum.no/index.php?board=120.0'}}

        constructeur = StringVar()
        modele = StringVar()

        Label(self, text="Constructeur automobile:", width=30).grid(row=1, sticky=N)

        constructeurs = StringVar()
        constructeurs.set(' '.join(list(cars.keys())))

        scrollbar_constructeurs = Scrollbar(self, orient=VERTICAL)
        lstbox_constructeurs = Listbox(self, listvariable=constructeurs, selectmode=SINGLE,
                                       yscrollcommand=scrollbar_constructeurs.set)
        scrollbar_constructeurs.config(command=lstbox_constructeurs.yview)
        lstbox_constructeurs.grid(row=1, column=1, sticky='NWS')
        scrollbar_constructeurs.grid(row=1, column=1, sticky='NESW', padx=125)

        def select():
            global constructeur
            selection = lstbox_constructeurs.curselection()
            constructeur = lstbox_constructeurs.get(selection)

            Label(self, text="Modèle:", width=30).grid(row=1, sticky=N)
            modeles = StringVar()
            modeles.set(' '.join(list(cars[constructeur].keys())))
            lstbox2 = Listbox(self, listvariable=modeles, selectmode=SINGLE)
            lstbox2.grid(row=1, column=2, sticky='NESW')

            def select2():
                global modele
                global link
                selection2 = lstbox2.curselection()
                modele = lstbox2.get(selection2)
                link = cars[constructeur][modele]

                def streaming(link, modele):

                    def Elbilforum():
                        cmd = 'cd elbilforum && cd elbilforum && cd && ' \
                              'echo {0} && ' \
                              'scrapy crawl cars -a link="{0}" -a car="{1}"' \
                            .format(link, modele)
                        os.system(cmd)

                    def start_foo_thread(event):
                        global foo_thread
                        foo_thread = threading.Thread(target=Elbilforum)
                        foo_thread.daemon = True
                        progressbar.start()
                        foo_thread.start()
                        self.after(20, check_foo_thread)

                    def check_foo_thread():
                        if foo_thread.is_alive():
                            self.after(20, check_foo_thread)
                        else:
                            progressbar.stop()
                            messagebox.showinfo("Info", "Scraping terminé.")
                            extra_window_streaming.destroy()

                    extra_window_streaming = Toplevel()
                    extra_window_streaming.geometry("+300+300")
                    extra_window_streaming.title("Avancement du scraping")
                    Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                    progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                    progressbar.grid()

                    ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                               command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                if link != "" and modele != "":
                    streaming(link, modele)

            btn = ttk.Button(self, text="Sélectionner", command=select2)
            btn.grid(row=2, column=2, sticky='NSW')

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.grid(row=2, column=1, sticky='NSW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=3, column=2, sticky='NESW')


class GoingElectric(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping de ElbilForum")

        cars = {'Audi': {'Audi_e-tron': './viewforum.php?f=150^&sid=776ae063bef89268a830ea09fe003080',
                         'Audi_A3_e-tron': './viewforum.php?f=108^&sid=776ae063bef89268a830ea09fe003080'},
                'BMW': {'BMW_i3': './viewforum.php?f=56^&sid=776ae063bef89268a830ea09fe003080'},
                'Hyundai': {'Hyundai_Ioniq_électrique': './viewforum.php?f=113^&sid=776ae063bef89268a830ea09fe003080',
                            'Hyundai_Kona_électrique': './viewforum.php?f=140^&sid=776ae063bef89268a830ea09fe003080',
                            'Hyundai_Ioniq_Hybride_rechargeable': './viewforum.php?f=143^&sid=776ae063bef89268a830ea09fe003080'},
                'Jaguar': {'Jaguar_i-Pace': './viewforum.php?f=149^&sid=776ae063bef89268a830ea09fe003080'},
                'KIA': {'Kia_Niro_EV': './viewforum.php?f=141^&sid=776ae063bef89268a830ea09fe003080',
                        'Kia_Soul_EV': './viewforum.php?f=95^&sid=776ae063bef89268a830ea09fe003080',
                        'KIA_Niro_hybride_rechargeable': './viewforum.php?f=142^&sid=776ae063bef89268a830ea09fe003080'},
                'Mercedes': {'Mercedes_Class_B': './viewforum.php?f=94^&sid=776ae063bef89268a830ea09fe003080',
                             'Mercedes_EQC': './viewforum.php?f=153^&sid=776ae063bef89268a830ea09fe003080'},
                'Mia_electric': {'Mia_electric': './viewforum.php?f=54^&sid=776ae063bef89268a830ea09fe003080'},
                'Nissan': {'Nissan_e-NV200': './viewforum.php?f=92^&sid=776ae063bef89268a830ea09fe003080',
                           'Nissan_Leaf': './viewforum.php?f=18^&sid=776ae063bef89268a830ea09fe003080'},
                'Opel': {'Opel_Ampera-e': './viewforum.php?f=114^&sid=776ae063bef89268a830ea09fe003080'},
                'Renault': {'Renault_Fluence_ZE.': './viewforum.php?f=21^&sid=776ae063bef89268a830ea09fe003080',
                            'Renault_Kangoo_ZE': './viewforum.php?f=23^&sid=776ae063bef89268a830ea09fe003080',
                            'Renault_Twizy': './viewforum.php?f=22^&sid=776ae063bef89268a830ea09fe003080',
                            'Renault_Zoe': './viewforum.php?f=20^&sid=776ae063bef89268a830ea09fe003080'},
                'Smart': {'smart_fortwo': './viewforum.php?f=132^&sid=776ae063bef89268a830ea09fe003080'},
                'Tesla': {'Tesla_Model_3': './viewforum.php?f=91^&sid=776ae063bef89268a830ea09fe003080',
                          'Tesla_Model_S': './viewforum.php?f=37^&sid=776ae063bef89268a830ea09fe003080',
                          'Tesla_Model_X': './viewforum.php?f=107^&sid=776ae063bef89268a830ea09fe003080'},
                'Volkswagen': {'Volkswagen_e-up!': './viewforum.php?f=50^&sid=776ae063bef89268a830ea09fe003080',
                               'Volkswagen_e-Golf': './viewforum.php?f=38^&sid=776ae063bef89268a830ea09fe003080',
                               'Volkswagen_Golf_GTE': './viewforum.php?f=111^&sid=776ae063bef89268a830ea09fe003080'},
                'Mitsubishi': {
                    'Mitsubishi_Plug-in_Hybrid_Outlander': './viewforum.php?f=109^&sid=776ae063bef89268a830ea09fe003080'}}

        constructeur = StringVar()
        modele = StringVar()

        Label(self, text="Constructeur automobile:", width=30).grid(row=1, sticky=N)

        constructeurs = StringVar()
        constructeurs.set(' '.join(list(cars.keys())))

        scrollbar_constructeurs = Scrollbar(self, orient=VERTICAL)
        lstbox_constructeurs = Listbox(self, listvariable=constructeurs, selectmode=SINGLE,
                                       yscrollcommand=scrollbar_constructeurs.set)
        scrollbar_constructeurs.config(command=lstbox_constructeurs.yview)
        lstbox_constructeurs.grid(row=1, column=1, sticky='NWS')
        scrollbar_constructeurs.grid(row=1, column=1, sticky='NESW', padx=125)

        def select():
            global constructeur
            selection = lstbox_constructeurs.curselection()
            constructeur = lstbox_constructeurs.get(selection)

            Label(self, text="Modèle:", width=30).grid(row=1, sticky=N)
            modeles = StringVar()
            modeles.set(' '.join(list(cars[constructeur].keys())))
            lstbox2 = Listbox(self, listvariable=modeles, selectmode=SINGLE)
            lstbox2.grid(row=1, column=2, sticky='NESW')

            def select2():
                global modele
                global link
                selection2 = lstbox2.curselection()
                modele = lstbox2.get(selection2)
                link = cars[constructeur][modele]

                def streaming(link, modele):

                    def Elbilforum():
                        cmd = 'cd goingelectric && cd goingelectric && cd && ' \
                              'echo {0} && ' \
                              'scrapy crawl cars -a url="{0}" -a car="{1}"' \
                            .format(link, modele)
                        os.system(cmd)

                    def start_foo_thread(event):
                        global foo_thread
                        foo_thread = threading.Thread(target=Elbilforum)
                        foo_thread.daemon = True
                        progressbar.start()
                        foo_thread.start()
                        self.after(20, check_foo_thread)

                    def check_foo_thread():
                        if foo_thread.is_alive():
                            self.after(20, check_foo_thread)
                        else:
                            progressbar.stop()
                            messagebox.showinfo("Info", "Scraping terminé.")
                            extra_window_streaming.destroy()

                    extra_window_streaming = Toplevel()
                    extra_window_streaming.geometry("+300+300")
                    extra_window_streaming.title("Avancement du scraping")
                    Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                    progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                    progressbar.grid()

                    ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                               command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                if link != "" and modele != "":
                    streaming(link, modele)

            btn = ttk.Button(self, text="Sélectionner", command=select2)
            btn.grid(row=2, column=2, sticky='NSW')

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.grid(row=2, column=1, sticky='NSW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=3, column=2, sticky='NESW')


class SpeakEV(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping de SpeakEV")

        cars = {'Training_Data': {'battery': 'https://www.speakev.com/search/38037/?q=range^&o=relevance/',
                                  'brakes': 'https://www.speakev.com/search/38184/?q=brakes^&o=relevance',
                                  'dealer': 'https://www.speakev.com/search/38001/?q=dealer^&o=relevance/',
                                  'engine_gearbox': 'https://www.speakev.com/search/38005/?q=engine+gearbox^&o=relevance/',
                                  'equipements': 'https://www.speakev.com/search/38010/?q=equipements+lights+connectivity^&o=relevance/',
                                  'interior_seats': 'https://www.speakev.com/search/38013/?q=interior+seats^&o=relevance/',
                                  'price': 'https://www.speakev.com/search/38015/?q=price^&o=relevance/'},
                'Audi': {'Audi_E-Tron_SUV': 'https://www.speakev.com/forums/audi-e-tron-suv.332/',
                         'General_Audi_EV_Discussion': 'https://www.speakev.com/forums/general-audi-ev-discussion.135/',
                         'Audi_A3_e-tron': 'https://www.speakev.com/forums/a3-etron/'},
                'BMW': {'General_BMW_EV_Discussion': 'https://www.speakev.com/forums/bmw-i/',
                        'BMW_i3': 'https://www.speakev.com/forums/bmw-i3/',
                        'BMW_i8': 'https://www.speakev.com/forums/bmw-i8/'},
                'Hyundai': {
                    'General_Hyundai_EV_Discussion': 'https://www.speakev.com/forums/general-hyundai-ev-discussion.169/',
                    'Hyundai_Ioniq_électrique': 'https://www.speakev.com/forums/first-generation-ioniq.177/',
                    'Hyundai_Kona_électrique': 'https://www.speakev.com/forums/hyundai-kona.281/'},
                'Mistsubishi': {'General_Mitsubishi_EV_Discussion': 'https://www.speakev.com/forums/mitsubishi-ev-gen/',
                                'Mitsubishi_Outlander_PHEV': 'https://www.speakev.com/forums/mitsubishi-evs/'},
                'Mercedes': {
                    'General_Mercedes_EV_Discussion': 'https://www.speakev.com/forums/general-mercedes-ev-discussion.144/'},
                'Nissan': {'General_Nissan_EV_Discussion': 'https://www.speakev.com/forums/nissan-evs/',
                           'Nissan_e-NV200': 'https://www.speakev.com/forums/nissan-env200/',
                           'Nissan_Leaf_2010-2017': 'https://www.speakev.com/forums/nissan-leaf24-leaf30.130/',
                           'Nissan_Leaf_2018': 'https://www.speakev.com/forums/nissan-leaf40.273/'},
                'KIA': {'General_Kia_EV_Discussion': 'https://www.speakev.com/forums/general-kia-ev-discussion.148/',
                        'Kia_Soul_EV': 'https://www.speakev.com/forums/soul-ev/',
                        'Kia_Niro_EV': 'https://www.speakev.com/forums/kia-niro.298/'},
                'Jaguar': {'General_JLR_Discussion': 'https://www.speakev.com/forums/general-jlr-discussion.201/',
                           'Jaguar_i-Pace': 'https://www.speakev.com/forums/jaguar-i-pace.209/'},
                'GM': {'General_GM_EV_Discussion': 'https://www.speakev.com/forums/gm-ev-discussion/',
                       'First_Generation_Ampera_and_Volt': 'https://www.speakev.com/forums/ampera-volt/'},
                'Renault': {'General_Renault_Z.E._Discussion': 'https://www.speakev.com/forums/renault-evs/',
                            'Renault_Zoe': 'https://www.speakev.com/forums/renault-zoe/',
                            'Renault_Twizy': 'https://www.speakev.com/forums/renault-twizy/'},
                'Volkswagen': {'General_Volkswagen_EV_Forum': 'https://www.speakev.com/forums/vw-evs/',
                               'Volkswagen_e-Golf': 'https://www.speakev.com/forums/vw-egolf/',
                               'Volkswagen_Golf_GTE': 'https://www.speakev.com/forums/vw-golf-GTE/',
                               'Volkswagen_Passat_GTE': 'https://www.speakev.com/forums/volkswagen-passat-gte.153/',
                               'Volkswagen_e-UP!': 'https://www.speakev.com/forums/vw-eup/',
                               'Volkswagen_ID.3': 'https://www.speakev.com/forums/volkswagen-id-3.316/'},
                'Tesla': {'General_Tesla_Motors_Discussion': 'https://www.speakev.com/forums/tesla-motors/',
                          'Tesla_Model_3': 'https://www.speakev.com/forums/tesla-model-3/',
                          'Tesla_Model_S': 'https://www.speakev.com/forums/tesla-model-s/',
                          'Tesla_Model_X': 'https://www.speakev.com/forums/tesla-model-x/',
                          'Tesla_Roadster': 'https://www.speakev.com/forums/tesla-roadster/'}}

        constructeur = StringVar()
        modele = StringVar()

        Label(self, text="Constructeur automobile:", width=30).grid(row=1, sticky=N)

        constructeurs = StringVar()
        constructeurs.set(' '.join(list(cars.keys())))

        scrollbar_constructeurs = Scrollbar(self, orient=VERTICAL)
        lstbox_constructeurs = Listbox(self, listvariable=constructeurs, selectmode=SINGLE,
                                       yscrollcommand=scrollbar_constructeurs.set)
        scrollbar_constructeurs.config(command=lstbox_constructeurs.yview)
        lstbox_constructeurs.grid(row=1, column=1, sticky='NWS')
        scrollbar_constructeurs.grid(row=1, column=1, sticky='NESW', padx=125)

        def select():
            global constructeur
            selection = lstbox_constructeurs.curselection()
            constructeur = lstbox_constructeurs.get(selection)

            Label(self, text="Modèle:", width=30).grid(row=1, sticky=N)
            modeles = StringVar()
            modeles.set(' '.join(list(cars[constructeur].keys())))
            lstbox2 = Listbox(self, listvariable=modeles, selectmode=SINGLE)
            lstbox2.grid(row=1, column=2, sticky='NESW')

            def select2():
                global modele
                global link
                selection2 = lstbox2.curselection()
                modele = lstbox2.get(selection2)
                link = cars[constructeur][modele]

                def streaming(link, modele):

                    def Elbilforum():
                        cmd = 'cd speakev && cd speakev && cd && ' \
                              'scrapy crawl cars -a url="{0}" -a car="{1}"' \
                            .format(link, modele)
                        os.system(cmd)

                    def start_foo_thread(event):
                        global foo_thread
                        foo_thread = threading.Thread(target=Elbilforum)
                        foo_thread.daemon = True
                        progressbar.start()
                        foo_thread.start()
                        self.after(20, check_foo_thread)

                    def check_foo_thread():
                        if foo_thread.is_alive():
                            self.after(20, check_foo_thread)
                        else:
                            progressbar.stop()
                            messagebox.showinfo("Info", "Scraping terminé.")
                            extra_window_streaming.destroy()

                    extra_window_streaming = Toplevel()
                    extra_window_streaming.geometry("+300+300")
                    extra_window_streaming.title("Avancement du scraping")
                    Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                    progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                    progressbar.grid()

                    ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                               command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                if link != "" and modele != "":
                    streaming(link, modele)

            btn = ttk.Button(self, text="Sélectionner", command=select2)
            btn.grid(row=2, column=2, sticky='NSW')

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.grid(row=2, column=1, sticky='NSW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=3, column=2, sticky='NESW')


class Caradisiac(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        Label(self, text="Scraping de Caradisiac")

        cars = {
            'Hyundai': {'Hyundai_Kona_électrique': 'http://forum-auto.caradisiac.com/marques/hyundai/debut-Kona.htm'},
            'Nissan': {'Nissan_Leaf': 'http://forum-auto.caradisiac.com/marques/nissan/debut-Leaf.htm'},
            'Renault': {'Renault_Zoe': 'http://forum-auto.caradisiac.com/marques/renault/debut-Zoe.htm',
                        'Renault_Twizy': 'http://forum-auto.caradisiac.com/marques/renault/debut-Twizy.htm'},
            'Tesla': {
                'General_Tesla_Motors_Discussion': 'http://forum-auto.caradisiac.com/marques/Autres-marques-Americaines/debut-tesla.htm'}}

        constructeur = StringVar()
        modele = StringVar()

        Label(self, text="Constructeur automobile:", width=30).grid(row=1, sticky=N)

        constructeurs = StringVar()
        constructeurs.set(' '.join(list(cars.keys())))

        scrollbar_constructeurs = Scrollbar(self, orient=VERTICAL)
        lstbox_constructeurs = Listbox(self, listvariable=constructeurs, selectmode=SINGLE,
                                       yscrollcommand=scrollbar_constructeurs.set)
        scrollbar_constructeurs.config(command=lstbox_constructeurs.yview)
        lstbox_constructeurs.grid(row=1, column=1, sticky='NWS')
        scrollbar_constructeurs.grid(row=1, column=1, sticky='NESW', padx=125)

        def select():
            global constructeur
            selection = lstbox_constructeurs.curselection()
            constructeur = lstbox_constructeurs.get(selection)

            Label(self, text="Modèle:", width=30).grid(row=1, sticky=N)
            modeles = StringVar()
            modeles.set(' '.join(list(cars[constructeur].keys())))
            lstbox2 = Listbox(self, listvariable=modeles, selectmode=SINGLE)
            lstbox2.grid(row=1, column=2, sticky='NESW')

            def select2():
                global modele
                global link
                selection2 = lstbox2.curselection()
                modele = lstbox2.get(selection2)
                link = cars[constructeur][modele]

                def streaming(link, modele):

                    def Caradisiac():
                        cmd = 'cd caradisiac && cd caradisiac && cd && ' \
                              'echo {0} && ' \
                              'scrapy crawl cars -a url="{0}" -a car="{1}"' \
                            .format(link, modele)
                        os.system(cmd)

                    def start_foo_thread(event):
                        global foo_thread
                        foo_thread = threading.Thread(target=Caradisiac)
                        foo_thread.daemon = True
                        progressbar.start()
                        foo_thread.start()
                        self.after(20, check_foo_thread)

                    def check_foo_thread():
                        if foo_thread.is_alive():
                            self.after(20, check_foo_thread)
                        else:
                            progressbar.stop()
                            messagebox.showinfo("Info", "Scraping terminé.")
                            extra_window_streaming.destroy()

                    extra_window_streaming = Toplevel()
                    extra_window_streaming.geometry("+300+300")
                    extra_window_streaming.title("Avancement du scraping")
                    Label(extra_window_streaming, text='Lancer le Scraping', width=100).grid(row=0)
                    progressbar = ttk.Progressbar(extra_window_streaming, orient='horizontal', mode='indeterminate')
                    progressbar.grid()

                    ttk.Button(extra_window_streaming, text="Commencer le Scraping",
                               command=lambda: start_foo_thread(None)).grid(column=0, row=1, sticky=E)

                if link != "" and modele != "":
                    streaming(link, modele)

            btn = ttk.Button(self, text="Sélectionner", command=select2)
            btn.grid(row=2, column=2, sticky='NSW')

        btn = ttk.Button(self, text="Sélectionner", command=select)
        btn.grid(row=2, column=1, sticky='NSW')

        button = Button(self, text="Page d'acceuil", command=lambda: controller.show_frame(StartPage))
        button.grid(row=3, column=2, sticky='NESW')


app = Scraping()
app.mainloop()
