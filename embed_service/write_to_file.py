import os

def write_list_to_file(file_name, header, values):
    out_dir=os.environ.get('OUTPUT_DIR','/var/log/thoth-ke/')
    with open(os.path.join(out_dir,f"{file_name}.txt"), "a") as _file:
        _file.write(f"\n## {header} ##\n")
        for val in values:
            _file.write(f"\n{val}\n")

def write_dict_to_file(file_name, header, _dict):
    out_dir=os.environ.get('OUTPUT_DIR','/var/log/thoth-ke/')
    with open(os.path.join(out_dir,f"{file_name}.txt"), "a") as _file:
        _file.write(f"\n## {header} ##\n")
        for _key, val in _dict.items():
            _file.write(f"\n{_key}={val}\n")

def write_dict_list_to_file(file_name, header, _dict):
    out_dir=os.environ.get('OUTPUT_DIR','/var/log/thoth-ke/')
    with open(os.path.join(out_dir,f"{file_name}.txt"), "a") as _file:
        _file.write(f"\n## {header} ##\n")
        for _key, values in _dict.items():
            _file.write(f"\n**** KEY - {_key} ****\n")
            for val in values:
                _file.write(f"\n- {val}\n")