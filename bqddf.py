import typing
import google
import eeddf
from google.cloud import bigquery

_BQ_CLIENT = None

_TEST_CONFIG = {
    "TABLE" : "eval_results",
    "DATASET" : "harness_test_db",
    "PROJECT_NAME" : "river-sky-386919",
}

_PROD_CONFIG = {
    "TABLE" : "eval_results",
    "DATASET" : "harness_prod_db",
    "PROJECT_NAME" : "timberidprd",
}

def get_config() -> typing.Dict[str, str]:
  if eeddf.is_test_environment():
    global _TEST_CONFIG
    return _TEST_CONFIG
  global _PROD_CONFIG
  return _PROD_CONFIG

def get_big_query_client() -> bigquery.Client:
  """
    Initializes and returns a connection to the BigQuery table.
  """

  global _BQ_CLIENT
  if not _BQ_CLIENT:

    # Make sure user is authenticated. They should call eeddf.initialize_ddf() or
    # some other authentication mechanism before establishing a BigQuery connection.
    credentials, _ = google.auth.default()
    if not credentials:
        return PermissionError(
            "You must authenticate yourself with your google cloud project before using this API.")
    
    _BQ_CLIENT = bigquery.Client(get_config()['PROJECT_NAME'])
  
  return _BQ_CLIENT

def get_eval_result(eval_id: str) -> bigquery.table.RowIterator:
  """
    Queries the experiment results table for eval information on the given id. Blocks
    until the query completes and returns the result. Returns a row iterator pointing
    to the beginning the results, but this iterator should continue just one value. 
  """
  client = get_big_query_client()

  # Set up SQL query
  table_name = f"{get_config()['DATASET']}.{get_config()['TABLE']}"
  query = f"SELECT * FROM {table_name} WHERE eval_id = '{eval_id}'"

  # Execute the query
  results = client.query_and_wait(query)
  if results.total_rows > 1:
    return ReferenceError(f"Two or more evals found for eval_id {eval_id}")
  return results

def insert_eval_result(eval: typing.Dict[str, typing.Any]) -> typing.List[typing.Dict[str, typing.Any]]:
  """
    Writes the eval result to the BigQuery table. Each key in the dict must correspond 
    to a column in the table's schema. Duplicate eval_ids can not be written. The dict's 
    values must also adhere to the table's schema. Returns a list of errors.
  """
  client = get_big_query_client()

  # Check if eval_id exists before writing it.
  exists = get_eval_result(eval['eval_id']).total_rows
  if exists:
    return [f"eval_id {eval['eval_id']} already exists"]

  # Set up reference to table we write to.
  table_ref = client.dataset(get_config()['DATASET']).table(get_config()['TABLE'])
  table = client.get_table(table_ref)

  # Automatically populate the timestamp field with the BigQuery commit time.
  eval['completion_timestamp'] = 'AUTO'

  # Insert data into BigQuery table.
  errors = client.insert_rows(table, [eval])
  return errors