from dataAnalysis import *

elink_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

pmid_list = read_pmids_from_file("PMIDs_list.txt")
all_gse_ids = []
all_gds_ids = []

for pmid in pmid_list:
    gse_ids, gds_ids = get_geo_ids(pmid, elink_url)

    all_gse_ids.extend(gse_ids)
    all_gds_ids.extend(gds_ids)

    # Print results for current PMID
    print(f"PMID {pmid} related to GEO ID:", gse_ids)

# Print all collected GSE and GDS IDs
print("All GSE IDs:", all_gse_ids)
print("All GDS IDs:", all_gds_ids)

# Fetch metadata for each GSE ID
metadata_list = [fetch_geo_metadata(gse_id) for gse_id in all_gse_ids]

for gse_id, metadata in zip(all_gse_ids, metadata_list):
    print(f"Metadane dla {gse_id}:\n", metadata)  # Podgląd

# Construct TF-IDF vectors
tfidf_matrix, feature_names = construct_tfidf_vectors(metadata_list)



# Check if there are enough samples for clustering
if tfidf_matrix.shape[0] < 3:
    print("Not enough samples for clustering. Adjusting n_clusters.")
    n_clusters = tfidf_matrix.shape[0]  # Set n_clusters to the number of samples
else:
    n_clusters = 3

# Step 3: Perform clustering
if n_clusters > 0:
    cluster_labels = cluster_datasets(tfidf_matrix, n_clusters=n_clusters)
    for i, gse_id in enumerate(all_gse_ids):
        print(f"GSE ID: {gse_id}, Cluster: {cluster_labels[i]}")
else:
    print("No samples available for clustering.")
