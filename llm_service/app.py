from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from constants import (
    MODEL_ID,
    MODEL_BASENAME,
    MODEL_ID_GPTQ,
    MODEL_BASENAME_GPTQ,
)
from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_gptq,
    load_full_model,
)

app = Flask(__name__)
device=os.getenv('DEVICE','cpu')
model_type=os.getenv('MODEL_TYPE',"gguf")
if model_type=="gptq":
    llm=load_full_model(MODEL_ID_GPTQ, MODEL_BASENAME_GPTQ, device, logging)
else:
    llm=load_quantized_model_gguf_ggml(MODEL_ID, MODEL_BASENAME, device, logging)
print("LLM loaded")
logging.info("LLM loaded")
recent_log=""


# Root path
@app.route('/')
def welcome():
    return f'Welcome to the AI Gateway! Use /call_llm with a json message. Model is {MODEL_BASENAME}.'


# call llm
@app.route('/call_llm', methods=['POST'])
def call_llm():
    try:
        data = request.json
        try:
            resp=llm(data["prompt"])
            logging.info(resp)
            return jsonify({'status': 'success', 'message': resp})
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, port=port)
