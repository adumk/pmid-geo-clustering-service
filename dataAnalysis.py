import numpy as np
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import silhouette_score


def read_pmids_from_file(filename: str) -> list:
    """
    Reads PMIDs from a text file.

    Args:
        filename (str): The name of the file containing PMIDs.

    Returns:
        list: A list of PMIDs read from the file.
    """
    try:
        with open(filename, 'r') as file:
            pmid_list = [line.strip() for line in file]
            return pmid_list
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def print_pmids_from_file():
    pmid_list = read_pmids_from_file("PMIDs_list.txt")
    if pmid_list:
        print("\n")
        print("List of PMIDs from file (copy them and paste in web service):\n", ", ".join(pmid_list))
        print("\n")
    else:
        print("No PMIDs in file")


def get_geo_ids(pmid: str, elink_url: str) -> list:
    """Pobiera GSE ID poprzez GDS na podstawie PMID."""
    params = {
        "dbfrom": "pubmed",
        "db": "gds",
        "id": pmid,
        "linkname": "pubmed_gds",
        "retmode": "xml",
    }
    response = requests.get(elink_url, params=params)

    if response.status_code != 200:
        print(f"Błąd HTTP {response.status_code} dla PMID {pmid}")
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        print(f"Błąd parsowania XML dla PMID {pmid}")
        return []

    gds_ids = []
    for link in root.findall(".//LinkSetDb/Link/Id"):
        gds_id = link.text
        gds_ids.append(gds_id)

    geo_ids = []
    for gds_id in gds_ids:
        gse_id = get_gse_id_from_gds(gds_id)
        if gse_id:
            geo_ids.append(gse_id)

    return geo_ids, gds_ids

def get_gse_id_from_gds(gds_id: str) -> str:
    """
    Retrieves the GEO Series (GSE) ID corresponding to a given GEO DataSet (GDS) ID.

    Args:
        gds_id (str): The GDS ID for which the GSE ID needs to be retrieved.

    Returns:
        str: The GSE ID if found, otherwise None.
    """
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    params = {
        "db": "gds",
        "id": gds_id,
        "retmode": "xml"
    }

    try:
        response = requests.get(esummary_url, params=params)

        if response.status_code != 200:
            print(f"HTTP Error {response.status_code} for GDS ID {gds_id}")
            return None

        root = ET.fromstring(response.content)

        gse_id = None
        for item in root.findall(".//Item[@Name='Accession']"):
            if item.text.startswith("GSE"):
                gse_id = item.text
                break

        return gse_id

    except ET.ParseError:
        print(f"XML Parsing Error for GDS ID {gds_id}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def fetch_geo_metadata(gse_id: str) -> dict:
    """
    Fetches metadata fields from a GEO dataset webpage.

    Args:
        gse_id (str): The GEO Series ID (e.g., "GSE15745").

    Returns:
        dict: A dictionary containing metadata fields (Title, Experiment type, Summary, Organism, Overall design).
    """
    base_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"Failed to fetch data for {gse_id}. HTTP Status Code: {response.status_code}")
        return {}

    soup = BeautifulSoup(response.content, "html.parser")

    metadata = {}

    metadata["Title"] = soup.find("td", text="Title").find_next_sibling("td").text.strip() if soup.find("td",
                                                                                                        text="Title") else ""
    metadata["Experiment type"] = soup.find("td", text="Experiment type").find_next_sibling(
        "td").text.strip() if soup.find("td", text="Experiment type") else ""
    metadata["Summary"] = soup.find("td", text="Summary").find_next_sibling("td").text.strip() if soup.find("td",
                                                                                                            text="Summary") else ""
    metadata["Overall design"] = soup.find("td", text="Overall design").find_next_sibling(
        "td").text.strip() if soup.find("td", text="Overall design") else ""

    metadata["Organism"] = soup.find("td", text="Organism").find_next_sibling(
        "td").text.strip() if soup.find("td", text="Organism") else ""

    if not metadata["Organism"]:
        organism_td = soup.find("td", text="Organisms")
        if organism_td:
            organism_cell = organism_td.find_next_sibling("td")
            organism_links = organism_cell.find_all('a')
            if organism_links:
                organisms_list = []
                for link in organism_links:
                    organism = link.get_text(strip=True)
                    organisms_list.append(organism)

            else:
                metadata["Organism"] = organism_cell.get_text(separator=" ").strip()
        else:
            metadata["Organism"] = ""
        metadata["Organism"] = "; ".join(organisms_list)

    return metadata

# def extract_organisms_from_samples(main_page_soup):
#     """
#     Extracts organisms from the samples section if 'Organism' field is empty.
#     """
#     organism_td = main_page_soup.find("td", text="Organism")
#     if organism_td:
#         organism_cell = organism_td.find_next_sibling("td")
#         links = organism_cell.find_all('a')
#         for link in links:
#             tekst = link.get_text(strip=True)
#             print(f"Organism: {tekst}\n")



def construct_tfidf_vectors(metadata_list: list) -> tuple:
    """
    Constructs TF-IDF vectors for a list of GEO dataset descriptions.

    Args:
        metadata_list (list): List of dictionaries containing metadata for multiple GEO datasets.

    Returns:
        tuple: TF-IDF matrix and feature names.
    """
    combined_texts = [
        " ".join(
            [meta.get(field, "") for field in ["Title", "Experiment type", "Summary", "Organism", "Overall design"]])
        for meta in metadata_list
    ]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    return tfidf_matrix, vectorizer.get_feature_names_out()

def cluster_datasets(tfidf_matrix, n_clusters=5):
    """
    Performs clustering on GEO dataset descriptions using K-Means.

    Args:
        tfidf_matrix: The TF-IDF matrix representing dataset descriptions.
        n_clusters (int): Number of clusters.

    Returns:
        list: Cluster labels for each dataset.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return cluster_labels


def optimal_clusters(tfidf_matrix, max_k=200):
    n_samples = tfidf_matrix.shape[0]

    max_k = min(max_k, n_samples - 1) if n_samples > 2 else 2
    if max_k < 2:
        return 1

    silhouette_scores = []

    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(tfidf_matrix)

            if len(set(labels)) > 1:
                score = silhouette_score(tfidf_matrix, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        except Exception as e:
            print(f"Error for k={k}: {str(e)}")
            silhouette_scores.append(0)

    if not silhouette_scores:
        return 1

    return np.argmax(silhouette_scores) + 2