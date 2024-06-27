import pandas_gbq
import ydata_profiling as pp
from google.oauth2 import service_account

# Assign variables for GCP project and credentials
gcp_project_id = "dce-gcp-training"

credentials = service_account.Credentials.from_service_account_file(
    '../idp_demand_forecasting/.config/gcp_service_account.json',
)

# Load data from BigQuery
products = pandas_gbq.read_gbq("SELECT * FROM `dce-gcp-training.idp_demand_forecasting.products`",
                           project_id=gcp_project_id, credentials=credentials)

print(pp.ProfileReport(products))
