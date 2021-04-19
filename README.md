# Sentiment: A Sentiment Analysis CLI Tool

## Run It Yourself
I strongly recommend that you use a python virtual environment. You can create one either using 
`venv`:
```
python3 -m venv name-of-virtual-env
source name-of-virtual-env/bin/activate
cd path/to/sentiment/
pip install -e .
````
Or `conda`:
```
conda create -n name-of-virtual-env 
```

After creating a virtual environment, the program `sentiment` should be available as a binary and 
installing the dependencies, and can be simply invoked using:
```
sentiment
```

For information about the options execute:
```
sentiment --help
```

