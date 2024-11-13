import os
import time
import nbformat
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
from dotenv import load_dotenv, set_key
from tqdm import tqdm
load_dotenv(override=True)

# Path to your .env file and Jupyter notebook
env_file = '../.env'
notebook_path = 'nsga_port_opt.ipynb'

# Function to update environment variables in the .env file
def update_env_variables(end_train_date, end_simulation_date, risk_measure, distribution):
    set_key(env_file, 'END_TRAIN_DATE', end_train_date)
    set_key(env_file, 'END_SIMULATION_DATE', end_simulation_date)
    set_key(env_file, 'RISK_MEASURE', risk_measure)
    set_key(env_file, 'DISTRIBUTION', distribution)

# Function to run the Jupyter notebook
def run_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    execute_preprocessor = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        execute_preprocessor.preprocess(nb, {'metadata': {'path': '.'}})
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                for output in cell.get('outputs', []):
                    # Only print text output (stdout)
                    if output.output_type == 'stream' and output.name == 'stdout':
                        print(output['text'])
        
        print(f"\nNotebook {notebook_path} executed successfully.")
        return True

    except Exception as e:
        print(f"Error executing the notebook: {e}")
        return False

# Loop over different configurations
configs = [
    {"end_train_date": "2021-11-01", "end_simulation_date": "2022-11-01", "risk_measure": "cvar", "distribution": "tstudent"}, ####### PERIOD 1 ########
    {"end_train_date": "2021-11-01", "end_simulation_date": "2022-11-01", "risk_measure": "var", "distribution": "normal"},
    {"end_train_date": "2021-11-01", "end_simulation_date": "2022-11-01", "risk_measure": "cvar", "distribution": "normal"},
    {"end_train_date": "2021-11-01", "end_simulation_date": "2022-11-01", "risk_measure": "var", "distribution": "tstudent"}, 
    {"end_train_date": "2007-06-01", "end_simulation_date": "2008-06-01", "risk_measure": "cvar", "distribution": "tstudent"}, ####### PERIOD 2 ########
    {"end_train_date": "2007-06-01", "end_simulation_date": "2008-06-01", "risk_measure": "var", "distribution": "normal"},
    {"end_train_date": "2007-06-01", "end_simulation_date": "2008-06-01", "risk_measure": "cvar", "distribution": "normal"},
    {"end_train_date": "2007-06-01", "end_simulation_date": "2008-06-01", "risk_measure": "var", "distribution": "tstudent"},
    {"end_train_date": "2022-11-01", "end_simulation_date": "2023-11-01", "risk_measure": "cvar", "distribution": "tstudent"}, ####### PERIOD 3 ########
    {"end_train_date": "2022-11-01", "end_simulation_date": "2023-11-01", "risk_measure": "var", "distribution": "normal"},
    {"end_train_date": "2022-11-01", "end_simulation_date": "2023-11-01", "risk_measure": "cvar", "distribution": "normal"},
    {"end_train_date": "2022-11-01", "end_simulation_date": "2023-11-01", "risk_measure": "var", "distribution": "tstudent"}
]

# Run the notebook for each configuration with a progress bar
with tqdm(total=len(configs), desc="Running Configurations") as pbar:
    start_time = time.time()
    for config in configs:
        print(f"\nRunning with config: {config}")
        
        # Update environment variables
        update_env_variables(
            config["end_train_date"],
            config["end_simulation_date"],
            config["risk_measure"],
            config["distribution"]
        )

        # Reload environment variables after updating .env file
        load_dotenv(override=True)

        # Run the notebook
        success = run_notebook(notebook_path)
        if success:
            print("Notebook executed successfully.")
        else:
            print("Notebook execution failed.")

        # Update progress bar
        pbar.update(1)

    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds.") 