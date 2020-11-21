#Importing libraries
import requests

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize , sent_tokenize 
import heapq
from flask import Flask, jsonify, request
import pickle 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Loading the models 
model = load_model("Sentiment_Analysis_on_TR")
with open('tokenize.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
app = Flask(__name__)

@app.route('/', methods = ['POST'])
def classify():
    Request = request.get_json()
    url = Request['url']
    id = Request['id']
    
    full_path = url + str(id) 
    response = requests.get(full_path)
    code = response.status_code
    if (code == 404):
        Data ={"Code:":code, "message":"Not found"}

    elif (code == 500):
        Data = {"Code:":code, "message":"Internal Server Error"}

    elif (code == 400):
        Data = {"Code:":code, "message":"Bad Request"}

    elif (code == 200):
        user = response.json()['blog']['userId']
        blog= response.json()['blog']
        
        content = blog['content']
        title = blog['title']
        author = user['fullName']
        Date_created = blog['createdAt'][0:10]
        Time_Created = blog['createdAt'][11:19]
        Last_Updated = blog['updatedAt'][0:10]
        Lime_Updated = blog['updatedAt'][11:19]

    
        text_ = content.lower()
        Stopwords = set(stopwords.words('english'))
        word_freq = {}
        for word in word_tokenize(text_):
            if word not in Stopwords:
                if word not in word_freq.keys():
                    word_freq[word] = 1 
                else:
                    word_freq[word] +=1
        max_freq = max(word_freq.values())
    
        for word in word_freq:
            word_freq[word] = word_freq[word]/max_freq
    
        sent_list = sent_tokenize(text_)
        sent_scores ={}
        for sent in sent_list:
            for word in word_tokenize(sent.lower()):
                if word in word_freq.keys():
                    if len(sent.split(' '))<30:
                        if sent not in sent_scores.keys():
                            sent_scores[sent] = word_freq[word]
                        else:
                            sent_scores[sent] += word_freq[word]
        summary_sentences = heapq.nlargest(10, sent_scores, key = sent_scores.get)
        Cap_sent=[]
        for tex in summary_sentences:
            tex_cap = tex.capitalize()
            Cap_sent.append(tex_cap)
        summary = " ".join(Cap_sent)
 
        text = content
        text_list = [text]
        text_token = tokenizer.texts_to_sequences(text_list)
        text_pad = pad_sequences(text_token, maxlen = 241, padding = 'pre')
        pred = model.predict(text_pad)
        if pred[0][0] > 0.5:
            result = "Positive Review!"
            per = round((pred[0][0])*100 , 2)

        else:
            result = 'Negative Review! '
            per = round((pred[0][0])*100,2) 
        Data = {'Title':title, 'Author':author,"Date_created":Date_created,
            'Time_Created':Time_Created, 'Summmary' : summary,
            'Last_Updated':Last_Updated, 'Lime_Updated':Lime_Updated,
            'Sentiment' : result, 'Certainity_Percentage': per, 'Contnt':content}
    elif (code == 401):
        Data ={"Code:":code, "message":"Unauthorized"}

    else: 
        Data ={"Code:":code, "message":"Not found"}
    return jsonify(Data)

if __name__ =='__main__':
     app.run(port = 800)