from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os
import pickle
import socket
from pymongo import MongoClient

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model_name = 'quora-distilbert-base'
model = SentenceTransformer(model_name)


@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("BERT", "world"), hostname=socket.gethostname())


@app.route("/update-vectors")
def updateVectors():
    try:
        save_path = './temp'
        client = MongoClient('mongodb://localhost:27017/')
        db = client.memoraiDevDB
        collections = db.list_collection_names()
        question_collections = []
        for collection in collections:
            if collection[:11] == "kbQuestions":
                question_collections.append(collection)
        for questions_collection in question_collections:
            file_name = questions_collection + ".pkl"
            complete_name = os.path.join(save_path, file_name)
            questions = []
            for question in db[questions_collection].find():
                if 'question' in question and question['isDuplicateQuestion'] is not True:
                    if 'answer' in question:
                        if question['kbQuestionStatus'] == 'ANSWER_APPROVED' or question['kbQuestionStatus'] == 'ADDED_FROM_VIEW_KB':
                            questions.append(question['question'])
            if len(questions) > 0:
                corpus_embeddings = model.encode(questions, show_progress_bar=True, convert_to_tensor=True)
            with open(complete_name, "wb") as fOut:
                pickle.dump({'sentences': questions, 'embeddings': corpus_embeddings}, fOut)
        return "true"
    except RuntimeError:
        return "false"


@app.route("/get-suggestions")
def getSuggestions():
    inp_question = request.args.get('q')
    organization_id = request.args.get('orgId')
    count = request.args.get("count")
    if count is None:
        count = 9
    file_name = "kbQuestions-" + organization_id + ".pkl"
    embedding_cache_path = os.path.join("./temp", file_name)
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences'][0:100000]
        corpus_embeddings = cache_data['embeddings'][0:100000]
        corpus_embeddings = corpus_embeddings.to(model._target_device)
        question_embedding = model.encode(inp_question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings)
        hits = hits[0]  # Get the hits for the first query
    suggestions = []
    for hit in hits[0:int(count)]:
        suggestion = {"content": corpus_sentences[hit['corpus_id']], "score": str(hit['score'])}
        suggestions.append(suggestion)
    return jsonify(suggestions)


if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')
