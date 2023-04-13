# ABOUT
This Project is based on Customer Purchase Behaviour Analysis and Prediction using RFM and Machine Learning Approach.

**The Project contains three main parts:**
1. Behaviour Analysis with RFM and Clustering with KMeans
2. Machine Learning Model Building (Naive Baye's, SVM, Random Forest, XGBoost and ANN)
3. Inference (Predition) App using Streamlit

```Parts 1 & 2``` are treated together as a single project involving Analysis and Model Building.
```Part 3``` is treated as a separate component

# DATA SOURCE
The dataset used is the Jewelry dataset obtained from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-jewelry-store).

# Folder Structure
- ```datasets```: Contains the dataset used for analysis and model training
- ```Training V1```: Contains all first training files (Notebooks, Results and Trained models)
- ```Training V2```: Contains all second training files (Notebooks, Results and Trained models)
- ```Training V3```: Contains all third (Final) training files(Notebooks, Results and Trained models)
    - ```results[x]```: Results obtained during training are saved here
    - ```trained_models[x]```: All trained models are saved in this folder
    - ```*.ipynb```: All Notebooks files for Analysis and Model Training
- ```SCREENSHOTS```: Some inference screenshots taken
- ```README.md```: Documentation markdown file containing Instructions on how to setup and access project contents
- ```requirements.txt```: Packages and project dependencies to be installed for inferencing.
- ```*.ipynb```: All Notebooks files for Analysis and Model Training

**Inference App:**
- ```InferenceApp```: The InferenceApp package files are here
- ```InferenceApp/app.py```: The main app file
- ```InferenceApp/myutils.py```: Contains some custom functions used by the ```app.py``` file

# Setup Prerequisites for the InferenceApp
- Goto Terminal (MAC) or Command Prompt (Windows)
- Navigate to project directory
- CD (change directory) into ```InferenceApp``` folder
```bash
cd InferenceApp
````

- Create a virtual Environment for 'streamlit-env' or choose any name of choice. Click [here - Official Python Docs](https://docs.python.org/3/library/venv.html) or [here - RealPython](https://realpython.com/python-virtual-environments-a-primer/), if you don't know how to do it.
```bash
$ python -m venv streamlit-env
```

- Activate the Virtual Environment

*On Windows:*
```bash
$ streamlit-env\Scripts\activate
```

*On MAC or LINUX:*
```bash
$ source streamlit-env/bin/activate
```

- Install Dependencies listed in the ```requirements.txt```file.
```bash
(streamlit-env) $ pip install -r requirements.txt
```


# Starting the Inference App (Running the App)
- Go to Terminal(MAC or Linux users) or Command Prompt (Windows users)
- Activate the ```streamlit-env``` Virtual Environment earlier created during setup, if not already activated.
- Navigate to Project Folder, if not already there.
- Navigate to ```InferenceApp``` folder, if not already there.
- Run the Inference App, while in the ```InferenceApp``` directory, run the command below:
```bash
(streamlit-env) $ streamlit run app.py
```

# Stopping the Running Inference App Server
- To terminate the Running App, go to the terminal and press ```Cntrl + C``` buttons
- Deactivate the Virtual Environment when done. Run:
```bash
(streamlit-env) $ deactivate
```
