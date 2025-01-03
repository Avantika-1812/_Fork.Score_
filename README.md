# ForkScore
ForkScore is an NLP and machine learning-based project designed to analyze and derive insights from reviews. Using techniques like sentiment analysis, topic modeling, and text classification, this tool helps in understanding user feedback, identifying trends, and providing recommendations for improvement.
## Run Locally
Clone the project

```bash
https://github.com/Avantika-1812/_Fork.Score_.git
```
Go to the project directory

```bash
cd _Fork.Score_
```
## Dataset Download

To download the project dataset run the following command in a terminal  

```bash
curl -o https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/input?select=Reviews.csv
```
Extract the zip file and change the folder name to Reviews_analysis.csv

## Resolving Dependencies
If using anaconda use the following command in base(root) terminal of anaconda to resolve all the depedencies
```bash
conda env create -f tf.yaml ```
```
```bash
conda activate tf ```
```
## How to run the Application
Open the review_analysis_app.py and run the followind command in the Terminal
```bash
python -m streamlit run review_analysis_app.py  ```
```
## Tech Stack
**Client:** Streamlit,HTML

**Server:** Python,Tensorflow,scikit-learn,numpy,pandas,matplotlib

## Running Tests

To run tests, Make sure current directory have the .keras file and class_indices.json file 

--> Run the model_ForkScore.ipynb file 
## FAQ

#### Question 1 Is this notebook GPU enabled ?

Answer 1 Yes

#### Question 2 Do i need GPU which supports CUDA 

Answer 2 Yes

