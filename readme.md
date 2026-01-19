
python3 -m venv venv
source venv/bin/activate
pip install "dvc[s3]"



dvc push data/iris.csv
dvc remote add -d s3remote s3://mlops-dvc-yash0707-prd