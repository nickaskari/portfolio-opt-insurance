import os
import json

# Function to append results to a JSON file
def append_to_json(file_path, new_data):
    # If the file exists, read its content, otherwise create an empty list
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new data to the list
    data.append(new_data)

    # Write updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)