import requests
import json
import pandas as pd
import time
import os
import csv

def get_new_key(login_name, login_password):
    login_request_URL = r"https://newapi.brandwatch.com/oauth/token"
    request_string = login_request_URL + "?username=" + login_name + "&password=" + login_password + r"&grant_type=api-password&client_id=brandwatch-api-client"
    response = requests.post(request_string)
    print(response.status_code)
    if response.status_code == 200:
        print("Acquire  key: success")
    else:
        print("Something went wrong")
        print(response.status_code)
        print(response.text)
    response_codes = json.loads(response.text)
    access_token = response_codes['access_token']
    return access_token


def list_projects(access_token):
    request_URL = "https://newapi.brandwatch.com/projects?access_token=" + str(access_token)
    response = requests.get(request_URL)
    if response.status_code == 200:
        print("Request: success")
    project_json = json.loads(response.text)
    project_list = pd.DataFrame(project_json['results'])
    print(project_list[['name', 'id']])
    return project_list


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
    request_URL = "https://newapi.brandwatch.com/projects/" + str(
        project_id) + "/queries/summary?access_token=" + access_token
    response = requests.get(request_URL)
    if response.status_code == 200:
        print("Get query ID: success")
    project_json = json.loads(response.text)
    project_summary = pd.DataFrame(project_json['results'])
    print(project_summary[['name', 'type', 'id']])
    return project_summary


def get_mentions_query_URL(start_date, end_date, project_id, query_id, access_token, cursor=False):
    query_def = "data/mentions"
    end_date = "endDate=" + end_date
    start_date = "startDate=" + start_date
    request_URL = "https://newapi.brandwatch.com/projects/" + str(project_id) + "/" + query_def
    request_URL = request_URL + "/fulltext"
    if cursor == False:
        request_URL = request_URL + "?" + "queryId=" + str(query_id ) + "&" + start_date + "&" + end_date + \
                      "&pageSize=5000" + "&access_token=" + access_token
    else:
        request_URL = request_URL + "?" + "queryId=" + str(query_id) + "&" + start_date + "&" + end_date +\
                      "&pageSize=5000" + "&cursor=" + str(cursor) + "&access_token=" + access_token
    return request_URL


def channel_query(query_def, start_date, end_date, project_id, query_id, access_token):
    '''
	project_json = channel_query(query_def,start_date,end_date,project_id,query_id,access_token)
	'''
    end_date = "?endDate=" + end_date
    start_date = "&startDate=" + start_date
    query_input = "&queryId=" + str(query_id)
    request_URL = "https://newapi.brandwatch.com/projects/" + project_id + "/" + query_def + end_date + query_input + start_date + "&access_token=" + access_token
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


def stream_data(file_name, date_debut, date_fin, project_id, query_id, access_token, new):
    bool = True
    cursor = False
    while bool == True:
        try:
            request_URL = get_mentions_query_URL(date_debut, date_fin, project_id, query_id, access_token, cursor)
            print(request_URL)
            mentions = get_mentions_data(request_URL)
        except:
            pass
        for i in range(len(mentions['results'])):
            with open(str(file_name), "a+") as file:
                writer = csv.writer(file)
                row = []
                col = []
                for key, value in mentions["results"][i].items():
                    row.append(str(value))
                if new == True and cursor == False and i == 0:
                    col = ["Resource Id","Account Type","Added","Assignment","Author","Author City","Author City Code","Author Continent","Author Continent Code",
                           "Author Country","Author Country Code", "Author Country","Author Country Code", "Author Location","Author State","Author State Code","Avatar Url","Average Duration Of Visit","Average Visits","backlinks","blogComments","categories","categoryDetails","checked",
                           "City","CityCode","Continent","ContinentCode","Country","Country Code","County","County Code","Date","Display Urls","Domain",
                           "engagement","expandedUrls","Facebook Author Id","Facebook Comments","Facebook Likes","Facebook Role","Facebook Shares",
                           "Facebook Subtype","ForumPosts","ForumViews","Full Text","Full name","gender","impact","importanceAmplification",
                           "ImportanceReach","Impressions","Influence","Insights Hashtag","Insights Mentioned","Instagram Comment Count",
                           "Instagram Follower Count","Instagram Following Count","Instagram Interactions Count","Instagram Like Count","Instagram Post Count",
                           "Interest","Language","Last Assignment Date","Latitude","Location Name","Longitude","Match Positions","Media Filter","Media Urls",
                           "Monthly Visitors","mozRank","Note Ids","Original Url","Outreach","Page Type","Pages Per Visit","Percent Female Visitors",
                           "Percent Male Visitors","Priority","Professions","Query Id","Query Name","Reach","Reach Estimate","Reply To","Resource Type",
                           "Twitter Retweet Of","Sentiment","Short Urls","Snippet","Starred","State","StateCode","Status","Subtype","Tags","Thread Author",
                           "Thread Created","Thread Entry Type","Thread Id","Thread URL","Title","Tracked Link Clicks","Tracked Links","Twitter Author Id",
                           "Twitter Followers","Twitter Following","Twitter PostCount","Twitter Reply Count","Twitter Retweets","Twitter Role","Twitter Verified",
                            "Updated","Url", "", "Classifier"]
                    col2 = [x for x in col]
                    writer.writerow(col2)
                writer.writerow(row)
        file.close()
        try:
            cursor = mentions['nextCursor']
        except:
            bool = False
