from flask import Flask, request, jsonify, redirect, url_for, render_template_string
from werkzeug.utils import secure_filename
import os
import logging
import ai_calls as ac 
from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    SOURCE_DIRECTORY,
    PERSIST_DIRECTORY,
)
ALLOWED_EXTENSIONS = ["csv","doc","docx","enex","epub","html","md","odt","pdf","ppt","pptx","txt"]

app = Flask(__name__)
device=os.getenv('DEVICE','cuda')
llm=ac.get_llm(MODEL_ID, MODEL_BASENAME, device, logging)
embeddings=ac.get_embeddings(EMBEDDING_MODEL_NAME,device,EMBEDDING_MODEL_PATH)
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
                    logging.error("unable to set db value")
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

def _allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Root path
@app.route('/')
def welcome():
    return 'Welcome to the AI Gateway! Use /process_documents, /call_llm, /cluster_docs with a json message.'

#SOURCE_DIRECTORY
@app.route('/upload_form')
def upload_form():
    return render_template_string('''
    <!doctype html>
    <title>Upload new Files</title>
    <h1>Upload new Files</h1>
    <form method=post enctype=multipart/form-data action="/upload">
      <label for="foldername">Folder Name:</label>
      <input type="text" id="foldername" name="foldername" required><br><br>
      <input type=file name=file[] multiple>
      <input type=submit value=Upload>
    </form>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    folder_name = request.form['foldername']
    if 'file[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('file[]')
    for file in files:
        if file and _allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create directory for the specified name if it doesn't exist
            target_folder = os.path.join(SOURCE_DIRECTORY, f"{folder_name}_docs")
            os.makedirs(target_folder, exist_ok=True)
            db_folder = os.path.join(PERSIST_DIRECTORY, f"{folder_name}_db")
            os.makedirs(db_folder, exist_ok=True)
            file.save(os.path.join(target_folder, filename))
    return redirect(url_for('upload_form'))

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
                logging.info(f"num_small_docs:{small_texts_cnt}, num_large_docs: {large_texts_cnt}")
                return jsonify({'status': 'success', 'message': {'num_small_docs':small_texts_cnt,'num_large_docs':large_texts_cnt}})
                #f"small and large chunks created for {data['path']}"})
            else:
                logging.error( "Unable to process docs")
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
            logging.info(resp)
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
            logging.info(resp)
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
                logging.info("clustered docs successfully")
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
                logging.info("db kv successfully loaded")
                return jsonify({'status': 'success', 'message': "db kv successfully loaded"})
            else:
                logging.error(recent_log)
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
                logging.info(qa_list)
                return jsonify({'status': 'success', 'message': qa_list})
            else:
                logging.error("get rag failed due to - db kv unable to load")
                return jsonify({'status': 'error', 'message': "get rag failed due to - db kv unable to load"})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, port=port)
