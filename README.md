Follow these steps to set up and run the project locally.

In PyCharm:
1. Clone or download the project from this github repository.
2. Open project in PyCharm and install required packages.
3. Run "app.py" file. You should see in terminal list of PMIDs from provided file and:

 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 522-323-947

Click the link `http://127.0.0.1:5000` in the PyCharm terminal to open the service in your browser.  
The terminal link may not be clickable in some configurations. Manually paste the URL into your browser if needed.
4. Insert one or more PMID in the web service (you can see an example there) and click submit. For datasets similar in size to the provided example file, the analysis may take **5–10 minutes**.

**View Results**  
   - After completion, the service will display:
     - Cluster visualizations.
     - PMID-GSE associations.
     - Detailed dataset information in a table format.
