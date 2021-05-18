# Download code and install environment

>>> conda env create -f environment.yml

# Activate environment

>>> conda activate thesis

# Install external dependencies not handled by environment

>>> python nltk
>>> nltk.download("all")

# Download and install java if not already installed

# Navigate to nlp_server/nlp_server/stanford-corenlp-4.1.0 and run api server (via powershell in Windowxs)

>>> java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

# Environment last refreshed: 13/05/2021

# This repo has large files, thus it is sometimes necessary to use github large file storage

# Make sure you install github lfs:

>>> git lfs install

# To save your repo using lfs, follow these steps:

>>> git lfs track "<large file name>"
>>> git add .gitattributes
>>> git commit -m "<your message>"
>>> git lfs migrate import --include="<large file here>"

# The above auto compresses the large files so they can be pushed. To pull the compressed data # back down in its original format:

>>> git pull
