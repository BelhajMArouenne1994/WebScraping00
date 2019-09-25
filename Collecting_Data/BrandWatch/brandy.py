import json
import re
import time

import pandas as pd
import pymongo
import requests
from mtranslate import translate
from pymongo import MongoClient


def replace_multiple(ch, s):
    new_str = ''.join([ch[i] for i in range(len(ch)-1) if (ch[i+1] != s or ch[i] != s)]+[ch[-1]])
    return new_str


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    cleantext = cleantext.replace("\r", ".").replace("\n", ".").replace("\t", ".").replace(";", ".")
    cleantext = replace_multiple(cleantext, " ")
    cleantext = replace_multiple(cleantext, ".")
    return cleantext


def get_new_key(login_name, login_password):
    login_request_URL = r"https://newapi.brandwatch.com/oauth/token"
    request_string = login_request_URL+"?username="+login_name+"&password="+login_password+r"&grant_type=api-password&client_id=brandwatch-api-client"
    response = requests.post(request_string)
    if response.status_code == 200:
        print("Acquire  key: success")
        response_codes = json.loads(response.text)
        access_token = response_codes['access_token']
    else:
        print("Something went wrong")
        access_token = "username-password incorrect(s)"
    return access_token


def list_projects(access_token):
    request_URL = "https://newapi.brandwatch.com/projects?access_token="+str(access_token)
    response = requests.get(request_URL)
    if response.status_code == 200:
        print("Request: success")
    project_json = json.loads(response.text)
    project_list = pd.DataFrame(project_json['results'])
    print(project_list[['name', 'id']])
    return (project_list[['name', 'id']])


def get_project_id_from_name(project_list, project_name):
    print(project_list[project_list['name'] == project_name][['name', 'id']])
    name = project_list[project_list['name'] == project_name][['id']].reset_index(drop=True)
    id = str(name['id'][0])
    return id


def get_query_id_from_name(query_list, query_name):
    print(query_list[query_list['name'] == query_name][['name', 'id']])
    name = query_list[query_list['name'] == query_name][['id']].reset_index(drop=True)
    id = str(name['id'][0])
    return id


def get_query_id(project_id, access_token):
    request_URL = "https://newapi.brandwatch.com/projects/"+str(
        project_id)+"/queries/summary?access_token="+access_token
    response = requests.get(request_URL)
    if response.status_code == 200:
        print("Get query ID: success")
    project_json = json.loads(response.text)
    print(project_json)
    project_summary = pd.DataFrame(project_json['results'])
    print(project_summary[['name', 'type', 'id']])
    return project_summary[['name', 'type', 'id']]


def get_mentions_query_URL(start_date, end_date, project_id, query_id, access_token, cursor=False):
    query_def = "data/mentions"
    end_date = "endDate="+end_date
    start_date = "startDate="+start_date
    request_URL = "https://newapi.brandwatch.com/projects/"+str(project_id)+"/"+query_def
    request_URL = request_URL+"/fulltext"
    if cursor == False:
        request_URL = request_URL+"?"+"queryId="+str(query_id)+"&"+start_date+"&"+end_date+ \
                      "&pageSize=1000"+"&access_token="+access_token+"&orderBy=date&orderDirection=asc"
    else:
        request_URL = request_URL+"?"+"queryId="+str(query_id)+"&"+start_date+"&"+end_date+ \
                      "&pageSize=1000"+"&cursor="+str(
            cursor)+"&access_token="+access_token+"&orderBy=date&orderDirection=asc"
    return request_URL


def channel_query(query_def, start_date, end_date, project_id, query_id, access_token):
    end_date = "?endDate="+end_date
    start_date = "&startDate="+start_date
    query_input = "&queryId="+str(query_id)
    request_URL = "https://newapi.brandwatch.com/projects/"+project_id+"/"+query_def+end_date+query_input+ \
                  start_date+"&access_token="+access_token
    print(request_URL)
    response = requests.get(request_URL)
    print(response.status_code)
    if response.status_code == 200:
        print("Acquire  key: success")
    else:
        print(response.text)
    project_json = json.loads(response.text)
    return project_json


def get_mentions_data(request_URL):
    response = requests.get(request_URL)
    print(response.status_code)
    if response.status_code != 200:
        print("Query: failure")
        print(request_URL)
        print(response.text)
    project_json = json.loads(response.text)
    return project_json


def boot_brandy(access_token):
    project_list = list_projects(access_token)
    return project_list


def stream_data(car_name, date_debut, date_fin, project_id, query_id, access_token, page_type_choosen):
    page_type_choosen = [pt.lower() for pt in page_type_choosen]
    client = MongoClient(port=27017)
    db = client.Renault
    state = ""
    bool = True
    cursor = False
    counter = 0
    duplicate = 0

    while bool == True:
        print(bool)
        try:
            request_URL = get_mentions_query_URL(date_debut, date_fin, project_id, query_id, access_token, cursor)
            print(request_URL)
            mentions = get_mentions_data(request_URL)
            if len(mentions['results']) == 0:
                bool = False
        except:
            bool = False
            pass

        if bool == True:
            try:
                cursor = mentions['nextCursor']
            except:
                bool = False

            for i in range(len(mentions['results'])):
                c = 0
                page_type = ""
                text = ""
                date = ""
                country = ""
                language = ""
                resource_Id = ""
                for key, value in mentions["results"][i].items():
                    if key == "fullText":
                        text = str(value)
                        c += 1
                    elif key == "date":
                        date = str(value)
                        c += 1
                    elif key == "country":
                        country = str(value)
                        c += 1
                    elif key == "language":
                        language = str(value)
                        c += 1
                    elif key == "pageType":
                        page_type = str(value)
                    elif key == "resourceId":
                        resource_Id = str(value)

                try:
                    if c == 4 and page_type in page_type_choosen:
                        counter += 1
                        if language == "en":
                            review = {
                                'ID': query_id+resource_Id,
                                'Car': car_name,
                                'Country': country,
                                'Text': cleanhtml(text),
                                'Date': date.split("T")[0],
                                'Page_Type': page_type[0].upper()+page_type[1:]
                            }
                            db.cars.ensure_index([('ID', pymongo.ASCENDING)], unique=True, dropDups=True)
                            dbItem = db.cars.find_one({'ID': query_id+resource_Id})
                            if dbItem:
                                counter -= 1
                                duplicate += 1
                            else:
                                db.cars.insert_one(review)
                        else:
                            try:
                                translated = translate(str(cleanhtml(text)), "en", language)
                            except:
                                print("wainting")
                                time.sleep(1)
                            review = {
                                'ID': query_id+resource_Id,
                                'Car': car_name,
                                'Country': country,
                                'Text': translated,
                                'Date': date.split("T")[0],
                                'Page_Type': page_type[0].upper()+page_type[1:]
                            }
                            db.cars.ensure_index([('ID', pymongo.ASCENDING)], unique=True, dropDups=True)
                            dbItem = db.cars.find_one({'ID': query_id+resource_Id})
                            if dbItem:
                                counter -= 1
                                duplicate += 1
                            else:
                                db.cars.insert_one(review)
                except:
                    pass

    try:
        state = str(counter)+' elements inserted.'+'\n'+str(duplicate)+" duplicates."
    except:
        pass

    print("end of streaming")
    print(state)
    return state
