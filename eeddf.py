import ee
import pandas as pd
import dataset
import raster
import eeraster
import importlib

importlib.reload(dataset)
importlib.reload(raster)
importlib.reload(eeraster)

from dataset import load_reference_samples
from eeraster import eeRaster
from raster import AmazonGeoTiffBase
from raster import stamp_isoscape
from eeraster import set_props
from eeraster import get_props
from eeraster import ingest_isoscape
from eeraster import dem
from eeraster import demXfab



_initialized = False
_test_environment = False
_project_name = 'river-sky-386919'

def initialize_ddf(test_environment:bool = False):
  '''
  Initializes the connection to the Ddf earth engine project. You will need to
  log into Earth Engine to succeed and you must have the proper authorization
  to access ddf specific data. Access is obtained by belonging to an
  organization added to TimberID.org.
  Once initialized with an environment, subsequent calls will not change the
  environment. You should call is_test_environment() to see which environmenbt
  has been loaded.
    Args:
      test_environment: If True, uses the Test environment and data. Defaults to False
  '''    
  global _initialized
  global _test_environment
  global _project_name
  if not _initialized:
    _test_environment = test_environment
    _initialized = True

    import google
    from google.colab import auth
    auth.authenticate_user()

    credentials, project_id = google.auth.default()
    if not _test_environment:
      _project_name = 'timberid-prd'

    ee.Initialize(credentials, project=_project_name, opt_url='https://earthengine-highvolume.googleapis.com')

def is_test_environment():
  '''
  Returns true if ddf was initialized with a test environment.
  '''
  global _test_environment
  return _test_environment

def ee_project_name():
  '''
  Returns the name of the EE project used for ddf
  '''
  return _project_name
