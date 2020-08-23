# Sentiment: A Naive Bayes-Based Sentiment Analysis CLI Tool
## TODO:
* File paths to models don't work, make path absolute/relative to script not where the user is (almost done, needs testing).
* Implement unit tests for model validation
* Train model using stopwords, FeatureHasher (able to implement online learning), balanced datasets + SVM.

## Run It Yourself
I strongly recommend that you use a python virtual environment. You can create one either using 
`venv`:
```
python3 -m venv name-of-virtual-env
source name-of-virtual-env/bin/activate
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

