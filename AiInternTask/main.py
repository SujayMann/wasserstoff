# Imports
import requests
import json
import os
import threading
import pymongo
import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import queue
from timeit import default_timer as timer
import psutil
from typing import List
from flask import Flask, request, jsonify

app = Flask(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Create 'downloads' folder to save downloaded pdfs
dir_name = 'downloads'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

# MongoDB connection
uri = os.getenv('MONGODB_URI')

# MongoDB Setup
myclient = pymongo.MongoClient(uri)
try:
    myclient.admin.command('ping')
    print("Connected to MongoDB.")

except Exception as e:
    print(e)

mydb = myclient['pdf_database']
mycol = mydb['pdfs']


# Semaphore
semaphore = threading.Semaphore(4)


# Store metadata in MongoDB collection
def store_metadata(key: str) -> None:
    """
    Stores id, name, filepath and size in a mongoDB collection

    Args:
        key (str): name of the PDF file
    """
    try:
        size = os.path.getsize(f"{dir_name}/{key}.pdf")
        mycol.insert_one({"_id": "_" + key, "name": key, "filepath": f'{dir_name}/{key}.pdf', "size": size})
    except Exception as e:
        print(f"Error in storing metadata for {key}: {e}")


# Download PDFs from URLs
def download_pdfs(url: str, key: str) -> None:
    """
    Downloads pdfs from given urls into a 'downloads' folder

    Args:
        url (str): url of the pdf source
        key (str): name of the pdf

    Returns:
        None
    """
    try:
        with semaphore:
            params = {'downloadformat': 'pdf'}
            response = requests.get(url, params)
            if response.status_code == 200:
                with open(f'{dir_name}/{key}.pdf', 'wb') as f:
                    f.write(response.content)
                    print(f"Successfully downloaded {key}.pdf")
                    store_metadata(key)
            else:
                print(f"Error occurred while downloading {key}: {response.status_code}")
    except Exception as e:
        print(f"Error occurred while fetching {url}: {e}")


# Threads
threads = []
content = json.load(open('Dataset.json', 'r'))

for key, value in content.items():
    t = threading.Thread(target=download_pdfs, args=(value, key))
    threads.append(t)
    t.start()

for t in threads:
    t.join()


# Summarizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Summarize text
def summarize(text: str, doc_length: int) -> str:
    """
    Generates summary for the given text from a pdf, based on the number of pages in the pdf using T5 transformer from hugging face.

    Args:
        text (str): text in the pdf
        doc_length (int): number of pages in the pdf

    Returns:
        summary (str): summary of the given text
    """
    max_len = 150
    if doc_length <= 10:
        max_len = 50
    elif 11 <= doc_length <= 30:
        max_len = 100
    
    ids = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(ids, max_length=max_len, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


# Extract keywords using TF-IDF
def extract_keywords(text: str) -> List:
    """
    Extracts 10 keywords from the given text using NLTK: TF-IDF Algorithm

    Args:
        text (str): text from a pdf

    Returns:
        List: containing of upto 10 keywords if generated, else []
    """
    try:
        tokens = word_tokenize(text)
        
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

        bigrams = list(nltk.bigrams(filtered_tokens))
        trigrams = list(nltk.trigrams(filtered_tokens))

        combined_tokens = filtered_tokens + [' '.join(bigram) for bigram in bigrams] + [' '.join(trigram) for trigram in trigrams]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(combined_tokens)])
        feature_array = tfidf_vectorizer.get_feature_names_out()
        tfidf_sorting = tfidf_matrix.sum(axis=0).A1.argsort()[::-1]
        
        top_keywords = [feature_array[i] for i in tfidf_sorting[:10]]
        return list(set(top_keywords))

    except Exception as e:
        print(f"Error while extracting keywords: {e}")
        return []


# Update MongoDB with summary and keywords
def update_data(id: str, summary: str, keywords: List) -> None:
    """
    Update MongoDB collection with summary and keywords extracted.

    Args:
        id (str): id of the pdf
        summary (str): summary of the pdf
        keywords (List): list containing the extracted keywords

    Returns:
        None
    """
    try:
        mycol.update_one(
            {'_id': "_" + id},
            {"$set": {'summary': summary, 'keywords': keywords}}
        )
    except Exception as e:
        print(f"Error in updating MongoDB for {id}: {e}")


# Process each PDF and update the DataBase
@app.route('/process', methods=['POST'])
def process_pdf(file_path: str):
    """
    Combines the functions for summarizing, extracting keywords and updating the database.
    Also handles any error occurred and logs time taken and memory consumed.

    Args:
        filepath: filepath of the pdf

    Returns:
        Status Code
    """
    start_time = timer()
    memory_before = psutil.Process().memory_info().rss
    
    try:
        with open(os.path.join(dir_name, file_path), 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            n_pages = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"

        summary = summarize(text, n_pages)
        keywords = extract_keywords(text)
        update_data(file_path.replace('.pdf', ''), summary, keywords)
        print(f"Completed processing {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    finally:
        end_time = timer()
        memory_after = psutil.Process().memory_info().rss
        total_time = end_time - start_time
        total_memory = abs(memory_after - memory_before) / (1024 * 1024) # convert to MB

        print(f"Processing Time for {file_path}: {total_time:.2f} seconds, Memory Used: {total_memory:.2f} MB")

    return jsonify({'status':'complete'}), 200

# PDF Queue
pdf_queue = queue.Queue()
for pdf_file in os.listdir(dir_name):
    if pdf_file.endswith('.pdf'):
        pdf_queue.put(pdf_file)


# # process pdfs
while not pdf_queue.empty():
    try:
        file_path = pdf_queue.get_nowait()
        with semaphore:
            process_pdf(file_path)
        pdf_queue.task_done()
    except queue.Empty:
        break

port = os.getenv('PORT')
app.run(host='0.0.0.0', port=port)
print("Done.")
