import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "detection_system"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/detector.py",
    f"{project_name}/alerts.py",
    f"{project_name}/utils.py",
    "config/config.json",
    "config/alert_config.json",
    "static/uploads/.gitkeep",
    "logs/.gitkeep",
    "models/.gitkeep",
    "templates/index.html",
    "app.py",
    "streamlit_app.py",
    "requirements.txt",
    "setup.sh",
    "config/config.yaml",
    ".dockerignore",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")