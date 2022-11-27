# ML Prediction API using LDA and NN for EEG Data

<pre> Built in Python with Flask for APIs & Deployed on Docker </pre>

## <strong><u>TLDR; {Build rightaway}</u></strong>

- Download this repo
  - <b>api.py</b> serves the API routes from Python-Flask application
  - <b>train.py</b> defines, builds and creates both NN and Classification Models and related model files
  - <b>requirements.txt</b> is a standard external packages dependency file for Python projects basically lists all 3rd party packages we use in the app
  - <b>Dockerfile</b> contains build, directory and environment info. Used to build an image of the total ML Prediction App that can be deployed as a container into any service.

### <strong>Build Docker Image</strong>

<code>docker build -t py-api -f Dockerfile .</code>

### <strong>Run built Docker Image in a Docker Container</strong>

<code>docker run -it -p 5000:5000 py-api python3 api.py</code>

---

### SOME CONTEXT

#### <strong>Goals</strong>

1. Build ML models to serve classifications and predictions using Flask API in Real Tine.
2. Deploy the model, package requirements for Python on a Docker Image
3. Test the image deployed on a container in localhost and on actual host on ports 5000

#### <strong>Test Inputs</strong>

1. test.json: 1300 rows of <a href="https://en.m.wikipedia.org/wiki/EEG_analysis">EEG</a> data; 160 features[columns]; Used to the test models
2. train.csv: Partial data to train the models

#### <strong>APIs with Desired Outcomes</strong>

- Data Extraction

    <small>
    Input: Row number <br>
    Output: Extract data and printed out to the console <br>
    Test Link: <a>http://127.0.0.1:5000/line/{lineNumber}</a>
    </small>

- Results from both models

    <small>
    Input: Row number <br>
    Process: Extract the selected row, inject new data into pre-trained and ddeployed models <br>
    Output: etrieve the classification prediction (Letter variable in the data)
    <br>
    Test Link : <a>http://127.0.0.1:5000/prediction/{lineNumber}</a>
    </small>

- Real-time Model Confidence Scores

    <small>
    Input: None <br>
    Process: Read all data from the local file {test.json} <br>
    Output: Print classification score of both the models. <br>

    Test Link: <a><http://127.0.0.1:5000/score></a>
    </snall>

---

### <u><i>Nice to know!</i></u>

- I used the idea from an article that outlined how Python-Flask-ML can be built and deployed on Docker
- Entirely coded, tested and deployed in Github Codespaces
- Output of ML may have extreme errors, which is not our focus, as we look to learn:
    1. How to develop a ML model on Python
    2. Make it accessible via Flask
    3. Package it as a Docker Image
    4. Deploy the built image as a container
- I did this in Github codespaces so had to test with in-browser terminal, outside http was not accsesible
- In the PORTS tab <i>[Next to TERMINAL tab in Codespaces]</i>, you can set the port to be exposed to public but not guaranteed of server access
- Major packages used:
    1. scikit-learn
    2. Flask
    3. Numpy

---

### TODO

- Model improvement
- Faster docker image build
- Smaller Image Layers
- Remove JSON based data storage
- Add DB access
- Check network, security, monitoring, infrastructure, orchestration in real PROD apps
