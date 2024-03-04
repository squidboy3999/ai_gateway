from flask import Flask, request, jsonify, redirect, url_for, render_template_string
from werkzeug.utils import secure_filename
import os
import logging
import doc_calls as dc 
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
processed_docs={}
recent_log=""

def _process_docs_helper(data):
    try:
        if not data['path'] in processed_docs.keys():
            small_texts, large_texts=dc.process_documents(data)
            processed_docs[data["path"]]={"small_texts":small_texts,"large_texts":large_texts}
        return True
    except Exception as e:
        logging.error(e)
        print(e)
        return False


def _allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Root path
@app.route('/')
def welcome():
    return 'Welcome to the Doc Handler Gateway! Use /process_documents, /cluster_docs with a json message.'

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

if __name__ == "__main__":
    port = int(os.getenv('PORT', '5006'))
    app.run(debug=True, port=port)
