import requests
from time import sleep
import os
import logging


def write_dict_to_file(_dict, header):
    out_dir=os.environ.get('OUTPUT_DIR','/var/log/ai_client/')
    with open(os.path.join(out_dir,f"ai_client.txt"), "a") as _file:
        _file.write(f"\n## {header} ##\n")
        for _key, val in _dict.items():
            _file.write(f"\n{_key}={val}\n")

# The URL of the server's endpoint
#http://127.0.0.1:5000/
url = f"{os.getenv('AI_GATEWAY','ai_gate')}:{int(os.getenv('PORT', '5000'))}/"
get_rag=f"http://{url}call_rag"
setup_db=f"http://{url}set_db_kv"
cluster_docs=f"http://{url}cluster_docs"
call_llm=f"http://{url}call_llm"
p_docs=f"http://{url}process_documents"
sleep_time=600
logging.info(f"Variables set sleeping for -{sleep_time}")

#wait for server to start up - LLM may take 10 minutes or more to boot up.
sleep(sleep_time)

# Sending a POST request to the server llm with prompt
response = requests.post(call_llm, json={'prompt': "You are a con man set out to trick the world into believing 1+1=5. Appeal to the deep inner needs that every person has. The reason 1+1=5 is"})
# Printing the response from the server
logging.info(response.json())
print(response.json())
write_dict_to_file(response.json(),"call_llm")

# Sending a POST request to the server to process docs
response = requests.post(p_docs, json={'path':"constitution_docs","small_chunk_size":1000,"small_chunk_overlap":200,"large_chunk_size":3000,"large_chunk_overlap":200})
# Printing the response from the server
logging.info(response.json())
print(response.json())
write_dict_to_file(response.json(),"process_documents")

# Sending a POST request to the server to process docs
response = requests.post(cluster_docs, json={'path':"constitution_docs","num_clusters":10,"cluster_samples":8})
# Printing the response from the server
logging.info(response.json())
print(response.json())
write_dict_to_file(response.json(),"cluster_docs")

# Sending a POST request to the server to setup vector db
response = requests.post(setup_db, json={"path":"constitution_docs","dir_name":"constitution_db"})
# Printing the response from the server
logging.info(response.json())
print(response.json())
write_dict_to_file(response.json(),"setup_db")

# Sending a POST request to the server to setup vector db - needs 'dir_name','path','questions','template'
_questions=["Am I allowed to own a gun as a law abiding citizen.","Can women vote?","Can troops stay in my home during peace time?"]
_template="""
Analyze the text segment provided and write a list of short answers and their reasons to the question.

 Your response should be in the form of a dictionary like a yaml format.

The dictionary should appear like the example below.
```
answer_dictionary:
    - answer: (a short answer to the question)
      answer_reason: (a short exaplantion of why this is an answer)
    - answer: (a short answer to the question)
      answer_reason: (a short exaplantion of why this is an answer) 
    ...

```
Given the following segment(s) provide a summary dictionary.

Question, additional info and Summary:
    |question|

Context Segment(s):
    |context|

answer_dictionary:
"""
response = requests.post(get_rag, json={"path":"constitution_docs",'dir_name':"constitution_db","questions":_questions,"template":_template})
# Printing the response from the server
logging.info(response.json())
print(response.json())
write_dict_to_file(response.json(),"get_rag")