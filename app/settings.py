import os

root_dir = os.path.dirname(os.path.abspath(__file__))
secret_path = os.path.join(os.path.abspath(os.path.join(root_dir, '..')), 'secret.yaml')
