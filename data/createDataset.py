import os
from dotenv import load_dotenv
import time
import json
import logging
from datetime import datetime
from typing import List, Dict

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
from Bio import Entrez
from bs4 import BeautifulSoup

# --------------------------- Configuration ---------------------------

# Set up logging
logging.basicConfig(
    filename='pubmed_scraper.log',  # Log file name
    level=logging.INFO,              # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv()

# Entrez Email (Required)
Entrez.email = os.getenv("ENTREZ_EMAIL")  # Replace with your actual email

# Entrez API Key (Optional but recommended)
Entrez.api_key = os.getenv("ENTREZ_API_KEY")  # Set as environment variable if available

# Rate Limiting
ENTREZ_RATE_LIMIT = 3  # Max 3 requests per second as per NCBI guidelines

# Sleep intervals based on rate limits
ENTREZ_SLEEP_INTERVAL = 1 / ENTREZ_RATE_LIMIT

# Batch size for fetching PubMed articles
PUBMED_BATCH_SIZE = 200

# Maximum number of articles to fetch
MAX_RESULTS = 1000  # Adjust as needed

# ScraperAPI Key (Set your actual ScraperAPI key here)
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")  # Replace with your ScraperAPI key
SCRAPER_API_BASE_URL = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url="

# PubMed EFetch base URL
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

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
        "retmax": max_results,
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

def extract_doi_from_pubmed(pmid):
    """
    Extract DOI from PubMed for the given PMID.
    """
    try:
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }

        logger.info(f"Fetching DOI for PMID: {pmid}")
        response = requests.get(PUBMED_FETCH_URL, params=fetch_params)
        response.raise_for_status()

        fetch_tree = ET.fromstring(response.content)
        doi = None

        # Extract DOI from MedlineCitation
        article_ids = fetch_tree.findall(".//ArticleIdList/ArticleId")
        for aid in article_ids:
            if aid.attrib.get("IdType") == "doi":
                doi = aid.text.strip()
                break

        if doi:
            logger.info(f"DOI for PMID {pmid}: {doi}")
        else:
            logger.warning(f"No DOI found for PMID {pmid}.")
        return doi
    except Exception as e:
        logger.error(f"Error extracting DOI for PMID {pmid}: {e}")
        return None

def fetch_citation_count(pmid):
    """
    Fetch the citation count for the given PMID from Google Scholar using ScraperAPI.
    """
    try:
        # Construct the Google Scholar query URL
        query_url = f"https://scholar.google.com/scholar?hl=en&q=pmid+{pmid}&btnG="
        scraper_url = f"{SCRAPER_API_BASE_URL}{query_url}"

        logger.info(f"Fetching citation count for PMID: {pmid}")
        response = requests.get(scraper_url, timeout=30)

        if response.status_code != 200:
            logger.error(f"Failed to fetch data for PMID {pmid}. HTTP {response.status_code}")
            return None

        # Parse the HTML response
        soup = BeautifulSoup(response.text, "html.parser")

        # Locate the "Cited by" link
        cited_by_link = soup.find("a", string=lambda text: text and text.startswith("Cited by"))
        if cited_by_link:
            # Extract the citation count
            citation_count = int(cited_by_link.text.split("Cited by")[1].strip())
            logger.info(f"PMID: {pmid} has {citation_count} citations.")
            return citation_count
        else:
            logger.warning(f"No citation count found for PMID {pmid}.")
            return 0
    except Exception as e:
        logger.error(f"Error fetching citation count for PMID {pmid}: {e}")
        return None

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

                    # Extract PMID
                    pmid_elem = medline_citation.find("PMID")
                    pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else ''
                    if not pmid:
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

                    # Extract DOI using the extract_doi_from_pubmed function
                    doi = extract_doi_from_pubmed(pmid)

                    # Fetch citation count using the fetch_citation_count function
                    citation_count = fetch_citation_count(pmid)

                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": abstract,
                        "journal": journal,
                        "doi": doi,
                        "citations": citation_count
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

    # Test mode flag
    test_mode = False  # Set to False to process all articles

    # Step 1: Search PubMed and fetch article details
    pmids = search_pubmed(query, max_results=1 if test_mode else MAX_RESULTS)
    if not pmids:
        logging.error("No PMIDs retrieved. Exiting.")
        return

    articles = fetch_pubmed_details(pmids)
    if not articles:
        logging.error("No article details fetched. Exiting.")
        return

    # Step 2: Save data to JSON
    output_filename = f'MS_SDoH_pubmed_abstracts_{datetime.now().strftime("%Y%m%d")}.json'
    save_to_json(articles, output_filename)

    logging.info("PubMed scraping and data collection completed successfully.")

if __name__ == "__main__":
    main()
