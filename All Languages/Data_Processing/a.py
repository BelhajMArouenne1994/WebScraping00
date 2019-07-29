def clean_forum_database(output_filename, input_filename, add=False, language = "english"):
    data_chunk = pd.read_csv(input_filename, chunksize= 10000, error_bad_lines=False, encoding='utf-8', delimiter=",")
    for i, data in enumerate(data_chunk):
        data.index = range(len(data.index))

        # processing reviews and counting words
        processedReviews = []
        lang = []
        rowsDelete = []

        if language != "english":
            for m in range(data.shape[0]):
                try:
                    data["Text"][m] = translate(data["Text"][m], "en", language)
                    print("+++++++++++++++++++++++++++++++++++++++")
                    print(m)
                    print("+++++++++++++++++++++++++++++++++++++++")
                    print(data["Text"][m])
                except:
                    print("waiting")
                    time.sleep(1)

        for m in range(data.shape[0]):
            data["Date"][m] = data["Date"][m].replace("[","").replace("]","").replace('"',"")
            data["Text"][m] = cleanhtml(data["Text"][m])
            data["Text"][m] = ''.join((x for x in data["Text"][m] if x not in punc))
            line = data["Text"][m].replace(".", ";").replace(",", ";").replace("r-link", "rlink").replace("r_link", "rlink").replace("r link", "rlink")
            # convert all urls to sting "URL"
            line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
            line = re.sub('pic[^\s]+', 'picture', line)
            line = re.sub('@[^\s]+', 'AT_USER', line)
            line = re.sub(r'#([^\s]+)', r'\1', line)
            tokens = word_tokenize(line)
            reviewProcessed = ''
            passer = False

            reviewProcessed=""
            for token in tokens:
                if token.lower() not in stopWords and passer == False and not all([c.isdigit() or c == '-' for c in token]):
                    token = stemmer.stem(token.lower())
                    reviewProcessed += token + " "
                    passer = False
            reviewProcessed = ''.join((x for x in reviewProcessed if x not in punc))
            reviewProcessed = reviewProcessed.replace("...", "")
            if len(reviewProcessed.split(" ")) > 10:
                processedReviews.append(reviewProcessed)
                lang.append(language)
            else:
                rowsDelete.append(m)
        # delting rows
        data.index = range(len(data.index))
        data = data.drop(rowsDelete, axis=0)
        data.index = range(len(data.index))
        data["processedReviews"] = processedReviews
        data["language"] = lang

        with open(output_filename + str(i) + ".csv", 'a+', encoding="utf-8") as file:
            data.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
            file.close()