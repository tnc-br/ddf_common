import typing
import google
import eeddf
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

_BQ_CLIENT = None

_TEST_CONFIG = {
    "METADATA_TABLE" : "eval_metadata",
    "PR_CURVE_TABLE" : "eval_pr_curves",
    "DATASET" : "harness_test_db",
    "PROJECT_NAME" : "river-sky-386919",
}

_PROD_CONFIG = {
    "METADATA_TABLE" : "eval_metadata",
    "PR_CURVE_TABLE" : "eval_pr_curves",
    "DATASET" : "harness_prod_db",
    "PROJECT_NAME" : "timberidprd",
}

def _get_config() -> typing.Dict[str, str]:
  if eeddf.is_test_environment():
    global _TEST_CONFIG
    return _TEST_CONFIG
  global _PROD_CONFIG
  return _PROD_CONFIG

def _get_big_query_client() -> bigquery.Client:
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
    
    _BQ_CLIENT = bigquery.Client(_get_config()['PROJECT_NAME'])
  
  return _BQ_CLIENT

def get_eval_result(eval_id: str) -> bigquery.table.RowIterator:
  """
    Queries the experiment results table for eval information on the given id. Blocks
    until the query completes and returns the result. Returns a row iterator pointing
    to the beginning the results, but this iterator should continue just one value. 
  """
  client = _get_big_query_client()

  # Set up SQL query
  table_name = f"{_get_config()['DATASET']}.{_get_config()['METADATA_TABLE']}"
  query = f"SELECT * FROM {table_name} WHERE eval_id = '{eval_id}'"

  # Execute the query
  results = client.query_and_wait(query)
  if results.total_rows > 1:
    return ReferenceError(f"Two or more evals found for eval_id {eval_id}")
  return results

def _insert_eval_metadata(metadata: typing.Dict[str, typing.Any]) -> typing.List[typing.Dict[str, typing.Any]]:
  client = _get_big_query_client()

  # Check if eval_id exists before writing it.
  exists = get_eval_result(metadata['eval_id']).total_rows
  if exists:
    raise GoogleAPIError(f"eval_id {metadata['eval_id']} already exists. " +
                          "An eval with these params has already run.")
  
  # Set up reference to table we write to.
  table_ref = client.dataset(_get_config()['DATASET']).table(_get_config()['METADATA_TABLE'])
  job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')

  # Write and block until complete.
  load_job = client.load_table_from_json(
    [metadata], table_ref, job_config=job_config)
  return load_job.result()

def _insert_eval_results(pr_curves: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[typing.Dict[str, typing.Any]]:
  client = _get_big_query_client()
  
  # Set up reference to table we write to.
  table_ref = client.dataset(_get_config()['DATASET']).table(_get_config()['PR_CURVE_TABLE'])
  job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')

  # Write and block until complete.
  load_job = client.load_table_from_json(
    pr_curves, table_ref, job_config=job_config)
  return load_job.result()

# Sort the keys and hash the printed result.
def _generate_eval_id(metadata: typing.Dict[str, typing.Any]) -> str:
  encoded = json.dumps(metadata, sort_keys=True)
  return "eval-" + str(hash(encoded))

def insert_eval(
    metadata: typing.Dict[str, typing.Any],
    results: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[typing.Dict[str, typing.Any]]:
  """
    Writes the eval result to the BigQuery table. Each key in the dict must correspond 
    to a column in the table's schema. Duplicate eval_ids can not be written. The dict's 
    values must also adhere to the table's schema. Returns a list of errors.

    This writes to two tables. Metadata is written to METADATA_TABLE and precision recall
    numbers are written to PR_CURVE_TABLE.

    Upon success, returns eval_id.
  """

  # Generate the eval_id and set it for metadata and PR curves.
  metadata['eval_id'] = _generate_eval_id(metadata)
  for result in results:
    result['eval_id'] = metadata['eval_id']

  # Write to metadata table
  metadata_write = _insert_eval_metadata(metadata)
  if (metadata_write.errors):
    raise GoogleAPIError(metadata_write.errors)
  
  # Write to PR curve table.
  result_write = _insert_eval_results(results)
  if (result_write.errors):
    raise GoogleAPIError(result_write.errors)
  
  return metadata['eval_id']