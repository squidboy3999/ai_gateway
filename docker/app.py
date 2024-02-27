from flask import Flask, request, jsonify
import os
import logging
import ai_calls as ac 
from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    EMBEDDING_MODEL_NAME,
)

app = Flask(__name__)
device=os.getenv('DEVICE','cpu')
llm=ac.get_llm(MODEL_ID, MODEL_BASENAME, device, logging)
embeddings=ac.get_embeddings(EMBEDDING_MODEL_NAME,'cpu',"/app/embed_model")
vector_dbs={}
processed_docs={}
recent_log=""

def _process_docs_helper(data):
    try:
        if not data['path'] in processed_docs.keys():
            small_texts, large_texts=ac.process_documents(data)
            processed_docs[data["path"]]={"small_texts":small_texts,"large_texts":large_texts}
        return True
    except Exception as e:
        logging.error(e)
        print(e)
        return False

def _set_db_kv_helper(data):
    if not data["dir_name"] in vector_dbs.keys():
        try:
            if _process_docs_helper(data):
                db, state=ac.get_doc_vectordb(data["dir_name"],processed_docs[data["path"]]["small_texts"],embeddings)
                recent_log=state
                if not (db is None):
                    vector_dbs[data["dir_name"]]=db
                    logging.info("vector db value set")
                    return True
                else:
                    return False
            else:
                logging.warning("Process docs helper failed")
                recent_log="Process docs helper failed"
                return False
        except Exception as e:
            logging.error(e)
            recent_log=f"Error - {e}"
            print(e)
            return False
    else:
        return True

# Root path
@app.route('/')
def welcome():
    return 'Welcome to the AI Gateway! Use /process_documents, /call_llm, /cluster_docs with a json message.'

# process_documents
@app.route('/process_documents', methods=['POST'])
def process_documents():
    try:
        data = request.json
        req_keys=['path',"small_chunk_size","small_chunk_overlap","large_chunk_size","large_chunk_overlap"]
        for req_key in req_keys:
            if not req_key in data.keys():
                return jsonify({'status': 'error', 'message': f"missing the {req_key} key"})
        try:
            # List contents of the folder
            #TODO, add state files for saving and loading information like this. State should be purgable as well.
            if _process_docs_helper(data):
                #processed_docs[data["path"]]
                small_texts_cnt=str(len(processed_docs[data["path"]]['small_texts']))
                large_texts_cnt=str(len(processed_docs[data["path"]]['large_texts']))
                return jsonify({'status': 'success', 'message': {'num_small_docs':small_texts_cnt,'num_large_docs':large_texts_cnt}})
                #f"small and large chunks created for {data['path']}"})
            else:
                return jsonify({'status': 'error', 'message': "Unable to process docs"})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

# run prompt on specific doc
@app.route('/doc_prompt', methods=['POST'])
def doc_prompt():
    try:
        # needs path, text_type, doc_number and template
        data = request.json
        try:
            text_docs=processed_docs[data['path']][data['text_type']][int(data['doc_number'])]
            doc_strings=ac.docs_to_strings(text_docs)
            resp=ac.call_llm(llm,data["template"].replace("|context|",doc_strings['context']))
            return jsonify({'status': 'success', 'message': {"response":resp,"context":doc_strings['context'], "metadata":doc_strings["metadata"]}})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

# call llm
@app.route('/call_llm', methods=['POST'])
def call_llm():
    try:
        data = request.json
        try:
            resp=ac.call_llm(llm,data["prompt"])
            return jsonify({'status': 'success', 'message': resp})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

# call get cluster docs
@app.route('/cluster_docs', methods=['POST'])
def cluster_docs():
    #TODO, can be done from list of docs sent in message, should also be done server side only.
    # data needs "texts","num_clusters","cluster_samples"
    try:
        data = request.json
        try: 
            if _process_docs_helper(data):
                tmp_data={}
                tmp_data["texts"]=processed_docs[data["path"]]["small_texts"]
                tmp_data['num_clusters']=data['num_clusters']
                tmp_data['cluster_samples']=data['cluster_samples']
                if not "clusters" in processed_docs[data["path"]].keys():
                    processed_docs[data["path"]]["clusters"]=ac.get_cluster_docs(embeddings,tmp_data)
                #processed_docs[data["path"]]["clusters"]=ac.get_cluster_docs(embeddings,data)
                #TODO return length of clusters if theses can be different than num_clusters
                return jsonify({'status': 'success', 'message': "clustered docs successfully"})
            else:
                return jsonify({'status': 'error', 'message': "processed docs failed"})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

# call set db key value
@app.route('/set_db_kv', methods=['POST'])
def set_db_kv():
    try:
        data = request.json
        try:
            if _set_db_kv_helper(data):
                return jsonify({'status': 'success', 'message': "db kv successfully loaded"})
            else:
                return jsonify({'status': 'error', 'message': recent_log})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

# call get_rag_qa_list
@app.route('/call_rag', methods=['POST'])
def call_rag():
    try:
        data = request.json
        try:
            if _set_db_kv_helper(data):
                qa_list=ac.get_rag_qa_list(llm,vector_dbs[data["dir_name"]],data['questions'],data['template'])
                return jsonify({'status': 'success', 'message': qa_list})
            else:
                return jsonify({'status': 'error', 'message': "get rag failed due to - db kv unable to load"})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, port=port)
