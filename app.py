import os

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

app = Flask(__name__)

API_KEY = "7aecff49d566458a9fc6dc4d6ba2c2c3"
RESOURCE_ENDPOINT = "https://us-jardiance-faqs.openai.azure.com"


openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

r = requests.get(url, headers={"api-key": API_KEY})

df=pd.read_csv(os.path.join(os.getcwd(),'envision_faq.csv'), sep=',')
df_faqs = df[['FAQ Request/Summary', 'FAQ Response', 'Request Category']]

pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_faqs['text']= df_faqs["FAQ Request/Summary"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_faqs['n_tokens'] = df_faqs["text"].apply(lambda x: len(tokenizer.encode(x)))
df_faqs = df_faqs[df_faqs.n_tokens<8192]

sample_encode = tokenizer.encode(df_faqs.text[0])
decode = tokenizer.decode_tokens_bytes(sample_encode)

df_faqs['ada_v2'] = df_faqs["text"].apply(lambda x : get_embedding(x, engine = 'vector-embedding'))

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="vector-embedding"
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res

@app.route('/')
def index():
    print('Running the vector embedding API')
    ques = request.args.get('ques')
    if ques is None:
        return render_template('index.html')
    res = search_docs(df_faqs, ques, top_n=4)
    data = res.loc[:,'FAQ Response'].values
    myDict = {}
    for i in range(0, len(data)):
        val = {i: data[i]}
        myDict.update(val)
    return jsonify(myDict)
#    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()
