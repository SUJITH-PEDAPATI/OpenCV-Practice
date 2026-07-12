from zipfile import ZipFile
import os
import splitfolders

def unzip_file(zip_path,extract_to):
    with ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(extract_to)

# unzip_file("./Day4 - Exploring to create YAML File/export.zip","./Day4 - Exploring to create YAML File/export")
splitfolders.ratio(
    input="./Day4 - Exploring to create YAML File/export",
    output="./Day4 - Exploring to create YAML File/dataset",
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)
