import json
import statistics
import matplotlib.pyplot as plt

# Load the JSON data from the file
with open('/Users/Tim/PycharmProjects/NII_Oxford_Project/MS_SDoH_pubmed_abstracts_20241127.json', 'r') as file:
    data = json.load(file)

# Initialize a counter and a list to store non-zero citation numbers
count = 0
citations_list = []

# Iterate through each entry in the JSON data
for entry in data:
    if entry.get('citations') and entry['citations'] > 0:
        count += 1
        citations_list.append(entry['citations'])

# Print the result
print(f"Number of entries with citations greater than 0: {count}")

# Calculate and print the median of the non-zero citation numbers
if citations_list:
    median_citations = statistics.median(citations_list)
    print(f"Median of non-zero citation numbers: {median_citations}")
else:
    print("No non-zero citation numbers found.")

# Initialize variables to track the highest citation count and corresponding PMID
max_citations = -1
max_citations_pmid = None

# Iterate through each entry in the JSON data
for entry in data:
    if entry.get('citations') is not None:
        citations = entry['citations']
        if citations > max_citations:
            max_citations = citations
            max_citations_pmid = entry['pmid']

# Print the result
print(f"PMID of the article with the highest citation number: {max_citations_pmid}")

# Remove the highest citation number from the list
if max_citations in citations_list:
    citations_list.remove(max_citations)

# Sort the citation numbers in decreasing order
citations_list.sort(reverse=True)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(citations_list)), citations_list)
plt.xlabel('Entries')
plt.ylabel('Number of Citations')
plt.title('Number of Citations in Decreasing Order (Excluding the Highest One)')
plt.show()