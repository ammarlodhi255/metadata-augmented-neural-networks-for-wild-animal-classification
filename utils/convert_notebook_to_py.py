import json

def notebook_to_script(notebook_file, script_file):
    # Read the Jupyter notebook file
    with open(notebook_file, 'r') as nb_file:
        notebook = json.load(nb_file)

    # Extract code cells from the Jupyter notebook
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']

    # Write the code from the code cells to the Python script file
    with open(script_file, 'w') as py_file:
        for cell in code_cells:
            for line in cell['source']:
                py_file.write(line)
            py_file.write('\n\n')

notebook_file = '/home/user-1/prog/masterthesis/src/py/temp.ipynb'
script_file = '/home/user-1/prog/masterthesis/src/py/viltkamera_classifier.py'

notebook_to_script(notebook_file, script_file)