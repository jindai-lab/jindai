import zipfile
from jindai import storage


def handle_zip(buf, *inner_path):
    """Handle zip file"""
    zpath = '/'.join(inner_path)
    with zipfile.ZipFile(buf, 'r') as zip_file:
        return zip_file.open(zpath)
    

storage.register_fragment_handler('zip', handle_zip)
