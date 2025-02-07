import typing
import google
import json
import eeddf
import datetime
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

_BQ_CLIENT = None

_CONFIG = {
    "METADATA_TABLE" : "eval_metadata",
    "PR_CURVE_TABLE" : "eval_pr_curves",
    "DATASET" : "harness_test_db",
    "TRAINING_METADATA_TABLE" : "training_runs",
    "FLATTENED_TABLE" : "harness_flattened_results",
    "PROJECT_NAME" : "river-sky-386919",
}

def _get_big_query_client() -> bigquery.Client:
  """
    Initializes and returns a connection to the BigQuery table.
  """

  global _BQ_CLIENT
  if not _BQ_CLIENT:
    _BQ_CLIENT = bigquery.Client(eeddf.ee_project_name())
  return _BQ_CLIENT

def get_eval_result(eval_id: str) -> bigquery.table.RowIterator:
  """
    Queries the experiment results table for eval information on the given id. Blocks
    until the query completes and returns the result. Returns a row iterator pointing
    to the beginning the results, but this iterator should continue just one value. 
  """
  client = _get_big_query_client()

  # Set up SQL query
  table_name = f"{_CONFIG['DATASET']}.{_CONFIG['METADATA_TABLE']}"
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
  table_ref = client.dataset(_CONFIG['DATASET']).table(_CONFIG['METADATA_TABLE'])
  job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')
  metadata['completion_timestamp'] = datetime.datetime.now().isoformat()

  # Write and block until complete.
  load_job = client.load_table_from_json(
    [metadata], table_ref, job_config=job_config)
  return load_job.result()

def _insert_eval_results(pr_curves: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[typing.Dict[str, typing.Any]]:
  client = _get_big_query_client()
  
  # Set up reference to table we write to.
  table_ref = client.dataset(_CONFIG['DATASET']).table(_CONFIG['PR_CURVE_TABLE'])
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

def get_training_result(training_id: str) -> bigquery.table.RowIterator:
  """
    Queries the experiment results table for training information on the given id. Blocks
    until the query completes and returns the result. Returns a row iterator pointing
    to the beginning the results, but this iterator should continue just one value. 
  """
  client = _get_big_query_client()

  # Set up SQL query
  table_name = f"{_CONFIG['DATASET']}.{_CONFIG['TRAINING_METADATA_TABLE']}"
  query = f"SELECT * FROM {table_name} WHERE training_id = '{training_id}'"

  # Execute the query
  results = client.query_and_wait(query)
  if results.total_rows > 1:
    return ReferenceError(f"Two or more trainings found for training_id {training_id}")
  return results

# Sort the keys and hash the printed result.
def _generate_training_id(metadata: typing.Dict[str, typing.Any]) -> str:
  encoded = json.dumps(metadata, sort_keys=True)
  return "training-" + str(hash(encoded))

def _insert_training_metadata(metadata: typing.Dict[str, typing.Any]) -> typing.List[typing.Dict[str, typing.Any]]:
  client = _get_big_query_client()

  # Check if eval_id exists before writing it.
  exists = get_training_result(metadata['training_id']).total_rows
  if exists:
    raise GoogleAPIError(f"training_id {metadata['training_id']} already exists. " +
                          "A training with these params has already run.")
  
  # Set up reference to table we write to.
  table_ref = client.dataset(_CONFIG['DATASET']).table(_CONFIG['TRAINING_METADATA_TABLE'])
  job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')

  metadata['completion_timestamp'] = datetime.datetime.now().isoformat()

  # Write and block until complete.
  load_job = client.load_table_from_json(
    [metadata], table_ref, job_config=job_config)
  return load_job.result()

def insert_training_metadata(
    metadata: typing.Dict[str, typing.Any]) -> typing.List[typing.Dict[str, typing.Any]]:
  """
    Writes the training metadata to the BigQuery table. Each key in the dict must correspond 
    to a column in the table's schema. Duplicate training_id can not be written. The dict's 
    values must also adhere to the table's schema. Returns a list of errors.

    This writes to TRAINING_METADATA_TABLE.

    Upon success, returns training_id.
  """

  # Generate the training_id and set it for training metadata.
  metadata['training_id'] = _generate_training_id(metadata)

  # Write to metadata table
  metadata_write = _insert_training_metadata(metadata)
  if (metadata_write.errors):
    raise GoogleAPIError(metadata_write.errors)
  
  return metadata['training_id']

def get_training_result_from_flattened(training_id: str) -> bigquery.table.RowIterator:
  """
    Queries the experiment results table for training information on the given id. Blocks
    until the query completes and returns the result. Returns a row iterator pointing
    to the beginning the results, but this iterator should continue just one value. 
  """
  client = _get_big_query_client()

  # Set up SQL query
  table_name = f"{_CONFIG['DATASET']}.{_CONFIG['FLATTENED_TABLE']}"
  query = f"SELECT * FROM {table_name} WHERE training_id = '{training_id}'"

  # Execute the query
  results = client.query_and_wait(query)
  return results

def get_eval_result_from_flattened(eval_id: str) -> bigquery.table.RowIterator:
  """
  Queries the flattened table for eval information on the given eval_id.

  Args:
      eval_id: The eval_id to search for.

  Returns:
      A RowIterator containing the results, or None if an error occurs.
      It's expected that the iterator will yield at most one row.
  """
  client = _get_big_query_client()
  table_name = f"{_CONFIG['DATASET']}.{_CONFIG['FLATTENED_TABLE']}"
  query = f"SELECT * FROM `{table_name}` WHERE eval_id = '{eval_id}'"

  results = client.query_and_wait(query)
  return results  


def insert_harness_run(
  training_metadata: typing.Dict[str, typing.Any],
  eval_results: typing.Dict[str, typing.Any],
  eval_only=False):
  """
  Writes training and eval results to flattened BQ table. 
  If the training_id already exists, don't overwrite it. 
  """

  client = _get_big_query_client()

  # Check if eval_id exists before writing it.
  exists = get_training_result_from_flattened(training_metadata['training_id']).total_rows
  if exists and not eval_only:
    raise GoogleAPIError(f"training_id {training_metadata['training_id']} already exists. " +
                          "A training with these params has already run.")

  eval_results['eval_id'] = _generate_eval_id(eval_results)
  flattened = training_metadata | eval_results
  flattened['completion_timestamp'] = datetime.datetime.now().isoformat()

  table_ref = client.dataset(_CONFIG['DATASET']).table(_CONFIG['FLATTENED_TABLE'])
  job_config = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')

  # Write and block until complete.
  load_job = client.load_table_from_json(
    [flattened], table_ref, job_config=job_config)
  return load_job.result()

def _check_exists(id_value: str, id_type: str):
  if id_type == 'training_id':
    exists = get_training_result_from_flattened(id_value).total_rows
  elif id_type == "eval_id":
    exists = get_eval_result_from_flattened(id_value)
  else: 
    raise GoogleAPIError(f"Unknown key type: {id_type}")

  if not exists:
    raise GoogleAPIError(f"{id_type} {id_value} does not exist. ")

def add_tag(id_value: str, tag: str, id_type: str = "training_id") -> None:
    """Adds a tag to the specified eval_id or training_id. If training_id is used, adds tag to 
    all rows with the specified training id.

    Args:
        id_value: The eval_id or training_id.
        tag: The tag to add.
        id_type: "eval_id" or "training_id" to specify the ID type. Defaults to "eval_id".
    """
    return _modify_tag(id_value, tag, add=True, id_type=id_type)  


def remove_tag(id_value: str, tag: str, id_type: str = "training_id") -> None:
    """Removes a tag from the specified eval_id or training_id. If training_id is used, adds tag to 
    all rows with the specified training id.

    Args:
        id_value: The eval_id or training_id.
        tag: The tag to remove.
        id_type: "eval_id" or "training_id" to specify the ID type. Defaults to "eval_id".
    """
    return _modify_tag(id_value, tag, add=False, id_type=id_type)


def _modify_tag(id_value: str, tag: str, add: bool, id_type: str = "eval_id") -> None:
    """Internal function to add or remove tags.  Used by add_tag and remove_tag."""

    _check_exists(id_value=id_value, id_type=id_type)
    client = _get_big_query_client()
    table_name = f"{_CONFIG['DATASET']}.{_CONFIG['FLATTENED_TABLE']}"

    if id_type not in ["eval_id", "training_id"]:
        raise ValueError("id_type must be 'eval_id' or 'training_id'")

    where_clause = f"{id_type} = '{id_value}'"

    if add:
        query = f"""
            UPDATE `{table_name}`
            SET tags = ARRAY_CONCAT(IFNULL(tags, []), ['{tag}'])
            WHERE {where_clause}
        """
        action = "added"
    else:  # Remove tag
        query = f"""
            UPDATE `{table_name}`
            SET tags = ARRAY(SELECT t FROM UNNEST(IFNULL(tags, [])) AS t WHERE t != '{tag}')
            WHERE {where_clause}
        """
        action = "removed"

    try:
        query_job = client.query(query)
        query_job.result()
        print(f"Tag '{tag}' {action} for {id_type} '{id_value}' successfully.")
    except Exception as e:
        print(f"Error modifying tag: {e}")
    
    # Query for the updated tags *after* the update
    get_tags_query = f"SELECT tags FROM `{table_name}` WHERE {where_clause}"
    get_tags_job = client.query(get_tags_query)
    get_tags_result = get_tags_job.result()
    return list(get_tags_result)
