import json

# Load the JSON data from the file
file_path = '/Users/Tim/PycharmProjects/NII_Oxford_Project/MS_SDoH_pubmed_abstracts_20241127.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the list of journals, avoiding duplicates
journals = {entry['journal'] for entry in data if 'journal' in entry}

# Print the list of journals
for journal in journals:
    print(journal)