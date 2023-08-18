import ee

_initialized = False

def initialize_ddf():
  '''
  Initializes the connection to the Ddf earth engine project. You will need to
  log into Earth Engine to succeed and you must have the proper authorization
  to access ddf specific data. Access is obtained by belonging to an
  organization added to TimberID.org
  '''    
  global _initialized
  if not _initialized:
    _initialized = True

    import google
    from google.colab import auth
    auth.authenticate_user()

    credentials, project_id = google.auth.default()
    ee.Initialize(credentials, project='river-sky-386919')
