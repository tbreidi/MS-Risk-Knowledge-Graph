import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
from Bio import Entrez

# --------------------------- Configuration ---------------------------

# Set up logging
logging.basicConfig(
    filename='pubmed_scraper.log',  # Log file name
    level=logging.INFO,              # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Entrez Email (Required)
Entrez.email = "your_email@example.com"  # Replace with your actual email

# Entrez API Key (Optional but recommended)
Entrez.api_key = os.getenv("ENTREZ_API_KEY")  # Set as environment variable if available

# Rate Limiting
ENTREZ_RATE_LIMIT = 3  # Max 3 requests per second as per NCBI guidelines
EUROPE_PMC_RATE_LIMIT = 10  # Max 10 requests per second as per Europe PMC guidelines
OPENALEX_RATE_LIMIT = 50  # Max 50 requests per second as per OpenAlex guidelines

# Sleep intervals based on rate limits
ENTREZ_SLEEP_INTERVAL = 1 / ENTREZ_RATE_LIMIT
EUROPE_PMC_SLEEP_INTERVAL = 1 / EUROPE_PMC_RATE_LIMIT
OPENALEX_SLEEP_INTERVAL = 1 / OPENALEX_RATE_LIMIT

# Batch size for fetching PubMed articles
PUBMED_BATCH_SIZE = 200

# Maximum number of articles to fetch
MAX_RESULTS = 525  # Adjust as needed

# OpenAlex API endpoint
OPENALEX_BASE_URL = "https://api.openalex.org/works/"

# ---------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 525) -> List[str]:
    """
    Searches PubMed for the given query and returns a list of PubMed IDs (PMIDs).
    """
    logging.info("Starting PubMed search...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,  # Set to desired number of results
        "retmode": "xml",
        "datetype": "pdat",
        "mindate": f"{datetime.now().year - 5}/01/01",
        "maxdate": f"{datetime.now().year}/12/31"
    }

    # Include Entrez API Key if available
    if Entrez.api_key:
        search_params["api_key"] = Entrez.api_key

    try:
        response = requests.get(base_url, params=search_params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during PubMed search request: {e}")
        return []

    try:
        search_tree = ET.fromstring(response.content)
        pmids = [id_elem.text for id_elem in search_tree.findall(".//Id")]
        total_articles = len(pmids)
        logging.info(f"Total articles found: {total_articles}")
        return pmids
    except ET.ParseError as e:
        logging.error(f"Error parsing PubMed search XML: {e}")
        return []

def fetch_pubmed_details(pmids: List[str]) -> List[Dict]:
    """
    Fetches detailed information for a list of PMIDs from PubMed.
    """
    articles = []
    total_pmids = len(pmids)
    logging.info("Fetching PubMed article details...")

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Initialize progress bar
    with tqdm(total=total_pmids, desc='Processing articles') as pbar:
        for i in range(0, total_pmids, PUBMED_BATCH_SIZE):
            batch_pmids = pmids[i:i + PUBMED_BATCH_SIZE]
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml"
            }

            # Include Entrez API Key if available
            if Entrez.api_key:
                fetch_params["api_key"] = Entrez.api_key

            try:
                response = requests.get(fetch_url, params=fetch_params)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error during PubMed fetch request: {e}")
                # Implement a short sleep before continuing
                time.sleep(5)
                pbar.update(len(batch_pmids))
                continue

            try:
                fetch_tree = ET.fromstring(response.content)
            except ET.ParseError as e:
                logging.error(f"Error parsing PubMed fetch XML: {e}")
                pbar.update(len(batch_pmids))
                continue

            # Extract abstracts and metadata
            for article in fetch_tree.findall(".//PubmedArticle"):
                try:
                    medline_citation = article.find("MedlineCitation")
                    if medline_citation is None:
                        pbar.update(1)
                        continue

                    # Extract Publication Year
                    pub_date = medline_citation.find(".//PubDate")
                    year_elem = pub_date.find("Year") if pub_date is not None else None
                    year = None
                    if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                        year = int(year_elem.text)
                    else:
                        # Try to extract year from MedlineDate
                        medline_date = pub_date.find("MedlineDate") if pub_date is not None else None
                        if medline_date is not None and medline_date.text:
                            year_str = medline_date.text.split()[0]
                            if year_str.isdigit():
                                year = int(year_str)

                    # Check publication types to exclude "Preprint"
                    pub_types = [ptype.text for ptype in medline_citation.findall(".//PublicationType")]
                    if "Preprint" in pub_types:
                        pbar.update(1)
                        continue

                    # Extract Title
                    title_elem = medline_citation.find(".//ArticleTitle")
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else 'No Title'

                    # Extract Authors
                    authors = []
                    for author in medline_citation.findall(".//Author"):
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        collective_name = author.find("CollectiveName")
                        if last_name is not None and fore_name is not None and fore_name.text and last_name.text:
                            authors.append(f"{fore_name.text.strip()} {last_name.text.strip()}")
                        elif collective_name is not None and collective_name.text:
                            authors.append(collective_name.text.strip())

                    # Extract Abstract
                    abstract = "No abstract available."
                    abstract_texts = [abstract_part.text.strip() for abstract_part in medline_citation.findall(".//Abstract/AbstractText") if abstract_part.text and abstract_part.text.strip()]
                    if abstract_texts:
                        abstract = " ".join(abstract_texts)

                    # Extract Journal Name
                    journal_elem = medline_citation.find(".//Journal/Title")
                    journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else 'No Journal Name'

                    # Extract DOI
                    doi = None
                    article_ids = medline_citation.findall(".//ArticleId")
                    for aid in article_ids:
                        if aid.attrib.get('IdType') == 'doi':
                            doi = aid.text.strip()
                            break

                    articles.append({
                        "pmid": medline_citation.findtext("PMID", default=""),
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": abstract,
                        "journal": journal,
                        "doi": doi,
                        "citations": None  # Placeholder for citation count
                    })

                    pbar.update(1)

                except AttributeError as e:
                    # Handle cases where an element is not found or does not have the expected structure
                    pmid = medline_citation.findtext("PMID", default="Unknown PMID")
                    logging.error(f"AttributeError processing PMID {pmid}: {e}")
                    pbar.update(1)
                    continue
                except Exception as e:
                    # Catch-all for any other exceptions
                    pmid = medline_citation.findtext("PMID", default="Unknown PMID")
                    logging.error(f"Error processing PMID {pmid}: {e}")
                    pbar.update(1)
                    continue

            # Respect Entrez rate limits
            time.sleep(ENTREZ_SLEEP_INTERVAL)

    logging.info(f"Total articles processed: {len(articles)}")

    return articles

def fetch_citation_counts(articles: List[Dict]) -> None:
    """
    Fetches citation counts from Europe PMC and OpenAlex for each article using PMID and DOI.
    Sets 'citations' to None if no citation count is available.
    """
    logging.info("Starting to fetch citation counts from Europe PMC and OpenAlex...")
    europe_pmc_base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    openalex_base_url = "https://api.openalex.org/works/"

    headers_europe_pmc = {
        "User-Agent": "PubMedScraper/1.0 (mailto:your_email@example.com)"  # Replace with your email
    }

    headers_openalex = {
        "User-Agent": "PubMedScraper/1.0 (mailto:your_email@example.com)",  # Replace with your email
        "Accept": "application/json"
    }

    with tqdm(total=len(articles), desc='Fetching citations') as pbar:
        for article in articles:
            citation_count = None
            pmid = article.get("pmid", "")
            doi = article.get("doi", "")

            # First attempt: Use OpenAlex if DOI is available
            if doi:
                # Normalize DOI by lowercasing and removing any whitespace
                doi_normalized = doi.lower().strip()
                api_url = f"{openalex_base_url}{doi_normalized}"

                try:
                    response = requests.get(api_url, headers=headers_openalex, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        citation_count = data.get('cited_by_count', 0)
                    elif response.status_code == 404:
                        # DOI not found in OpenAlex
                        citation_count = 0
                    else:
                        logging.error(f"OpenAlex API Error {response.status_code} for DOI {doi}: {response.text}")
                        citation_count = None
                except requests.exceptions.RequestException as e:
                    logging.error(f"Error fetching citation count from OpenAlex for DOI {doi}: {e}")
                    citation_count = None

            # Second attempt: Use Europe PMC if DOI is not available or OpenAlex failed
            if citation_count is None and pmid:
                # Construct the query to search for citations of the given PMID
                # Europe PMC uses "REF_ID:{pmid}" to find articles that cite the given PMID
                query = f"REF_ID:{pmid}"
                params = {
                    "query": query,
                    "format": "json",
                    "resulttype": "core",
                    "pageSize": 0  # We only need the total count
                }

                try:
                    response = requests.get(europe_pmc_base_url, params=params, headers=headers_europe_pmc, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    citation_count = data.get('hitCount', 0)
                except requests.exceptions.RequestException as e:
                    logging.error(f"Error fetching citation count from Europe PMC for PMID {pmid}: {e}")
                    citation_count = None
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON response from Europe PMC for PMID {pmid}: {e}")
                    citation_count = None

            # Assign the citation count
            article['citations'] = citation_count

            # Respect rate limits
            if doi:
                time.sleep(OPENALEX_SLEEP_INTERVAL)
            elif pmid:
                time.sleep(EUROPE_PMC_SLEEP_INTERVAL)
            else:
                # If neither DOI nor PMID is available, minimal sleep
                time.sleep(0.1)

            pbar.update(1)

    logging.info("Completed fetching citation counts from Europe PMC and OpenAlex.")

def save_to_json(data: List[Dict], filename: str) -> None:
    """
    Saves the data to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to JSON: {e}")

def main():
    # Define your search query
    query = '"multiple sclerosis" AND ("social determinants" OR "environmental factors")'

    # Step 1: Search PubMed and fetch article details
    pmids = search_pubmed(query, max_results=MAX_RESULTS)
    if not pmids:
        logging.error("No PMIDs retrieved. Exiting.")
        return

    articles = fetch_pubmed_details(pmids)
    if not articles:
        logging.error("No article details fetched. Exiting.")
        return

    # Step 2: Fetch citation counts from OpenAlex and Europe PMC
    fetch_citation_counts(articles)

    # Step 3: Save data to JSON
    output_filename = f'MS_SDoH_pubmed_abstracts_{datetime.now().strftime("%Y%m%d")}.json'
    save_to_json(articles, output_filename)

    logging.info("PubMed scraping and data collection completed successfully.")

if __name__ == "__main__":
    main()
