import ee
from rich import print


def authenticate_gee(project: str):
    """Authenticate to Google Earth Engine"""
    try:
        ee.Initialize(project=project)
        print("GEE authentication successful!")
    except ee.ee_exception.EEException:
        ee.Authenticate(quiet=False, auth_mode="localhost")
        ee.Initialize(project=project)
        print("GEE authentication completed!")
