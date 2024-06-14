from sentence_transformers import SentenceTransformer, evaluation
from collections import defaultdict
from sentence_transformers.readers import InputExample
import csv
import os
import random

model_name = 'quora-distilbert-base'
model = SentenceTransformer(model_name)


def make_rows():
    sentences = {}
    duplicates = defaultdict(lambda: defaultdict(bool))
    corpus_ids = set()
    with open("./responsify.tsv", encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            id1 = row['q1id']
            id2 = row['q2id']
            corpus_ids.add(id1)
            corpus_ids.add(id2)
            question1 = row['q1'].replace("\r", "").replace("\n", " ").replace("\t", " ")
            question2 = row['q2'].replace("\r", "").replace("\n", " ").replace("\t", " ")
            is_duplicate = row['is_duplicate']

            if question1 == "" or question2 == "":
                continue

            sentences[id1] = question1
            sentences[id2] = question2

            if is_duplicate == '1':
                duplicates[id1][id2] = True
                duplicates[id2][id1] = True
    return [sentences, duplicates, corpus_ids]


def make_corpus(corpus_ids, sentences):
    with open('corpus.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid\tquestion\n")
        for id in sorted(list(corpus_ids), key=lambda id: int(id)):
            fOut.write("{}\t{}\n".format(id, sentences[id]))


def make_ir(corpus_ids, sentences, duplicates):
    with open('test-queries.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid\tquestion\tduplicate_qids\n")
        for id in sorted(list(corpus_ids), key=lambda id: int(id)):
            fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))


def evaluate_model():
    sentences, duplicates, corpus_ids = make_rows()
    make_corpus(corpus_ids, sentences)
    make_ir(corpus_ids, sentences, duplicates)
    max_corpus_size = 100
    ir_queries = {}
    ir_needed_qids = set()
    ir_corpus = {}
    ir_relevant_docs = {}
    with open(os.path.join('test-queries.tsv'), encoding='utf8') as fIn:
        next(fIn)  # Skip header
        for index, line in enumerate(fIn):
            if len(line.strip().split('\t')) < 3:
                continue
            qid, question, duplicate_ids = line.strip().split('\t')
            duplicate_ids = duplicate_ids.split(',')
            ir_queries[qid] = question
            ir_relevant_docs[qid] = set(duplicate_ids)
            for qid in duplicate_ids:
                ir_needed_qids.add(qid)
        # First get all needed relevant documents (i.e., we must ensure, that the relevant questions are actually in
        # the corpus
        distraction_questions = {}
        with open(os.path.join('corpus.tsv'), encoding='utf8') as fIn:
            next(fIn)  # Skip header
            for line in fIn:
                if len(line.strip().split('\t')) < 2:
                    continue
                qid, question = line.strip().split('\t')
                if qid in ir_needed_qids:
                    ir_corpus[qid] = question
                else:
                    distraction_questions[qid] = question
        # Now, also add some irrelevant questions to fill our corpus
        other_qid_list = list(distraction_questions.keys())
        random.shuffle(other_qid_list)

        for qid in other_qid_list[0:max(0, max_corpus_size - len(ir_corpus))]:
            ir_corpus[qid] = distraction_questions[qid]

        # get test samples
        test_samples = []
        with open(os.path.join("responsify.tsv"), encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                sample = InputExample(texts=[row['q1'], row['q2']], label=int(row['is_duplicate']))
                test_samples.append(sample)

        evaluators = []
        ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)
        evaluators.append(ir_evaluator)
        seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
        model.evaluate(evaluator=seq_evaluator, output_path="output/evaluation")
