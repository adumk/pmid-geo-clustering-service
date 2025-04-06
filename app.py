from flask import Flask, request, render_template
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from dataAnalysis import (
    read_pmids_from_file,
    print_pmids_from_file,
    optimal_clusters,
    get_geo_ids,
    fetch_geo_metadata,
    construct_tfidf_vectors,
    cluster_datasets
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pmids_input = request.form['pmids']
        pmids = [pmid.strip() for pmid in pmids_input.split(',') if pmid.strip()]

        all_gse_ids = []
        metadata_list = []
        pmid_to_gse = {}

        for pmid in pmids:
            try:
                gse_ids, _ = get_geo_ids(pmid, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi")
                all_gse_ids.extend(gse_ids)
                pmid_to_gse[pmid] = gse_ids

                for gse_id in gse_ids:
                    try:
                        metadata = fetch_geo_metadata(gse_id)
                        if metadata:  # Pomijaj puste metadane
                            metadata_list.append({
                                'gse_id': gse_id,
                                'pmid': pmid,
                                **metadata
                            })
                    except Exception as e:
                        print(f"Error fetching metadata for {gse_id}: {str(e)}")
            except Exception as e:
                print(f"Error processing PMID {pmid}: {str(e)}")

        df = pd.DataFrame(metadata_list)

        if len(df) >= 1:
            try:
                combined_texts = [
                    " ".join([str(meta.get(field, "")) for field in
                              ["Title", "Experiment type", "Summary", "Organism", "Overall design"]])
                    for meta in metadata_list
                ]
                if any(text.strip() for text in combined_texts):
                    tfidf_matrix, _ = construct_tfidf_vectors(metadata_list)
                else:
                    raise ValueError("All metadata fields are empty")
                if len(df) == 1:
                    df['cluster'] = 0
                    df['x'] = [0.0]
                    df['y'] = [0.0]

                elif len(df) == 2:
                    df['cluster'] = [0, 1]
                    df['x'] = [0.0, 1.0]
                    df['y'] = [0.0, 1.0]

                else:
                    n_clusters = min(
                        optimal_clusters(tfidf_matrix),
                        len(df) - 1,
                        200  # max 200 clusters
                    )
                    n_clusters = max(n_clusters, 2) # min 2 clusters

                    df['cluster'] = cluster_datasets(tfidf_matrix, n_clusters=n_clusters)
                    n_samples = len(df)
                    perplexity = max(1, min(30, n_samples - 1))

                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        random_state=42,
                        n_iter=1000,
                        init='pca'
                    )

                    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
                    df['x'] = tsne_results[:, 0].astype(float)
                    df['y'] = tsne_results[:, 1].astype(float)

            except Exception as e:
                print(f"Critical error in processing pipeline: {str(e)}")
                df['cluster'] = 0
                df['x'] = np.linspace(0, 1, len(df)).tolist()
                df['y'] = np.zeros(len(df)).tolist()

                df = df.fillna('')

            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='cluster',
                hover_data=['gse_id', 'pmid', 'Title', 'Organism'],
                title='Clusterization of GEO datasets'
            )
            plot_html = fig.to_html(full_html=False)
        else:
            plot_html = None

        return render_template(
            'results.html',
            plot_html=plot_html,
            pmid_to_gse=pmid_to_gse,
            df=df.to_dict('records')
        )

    return render_template('index.html')

if __name__ == '__main__':
    import os
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print_pmids_from_file()
    app.run(debug=True)