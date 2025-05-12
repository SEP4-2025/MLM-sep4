# MLM-sep4

This is a repository with all the code for the SEP4 backend teams Machine learning model. It gets data from backend and returns future predictions

How to setup the environment right:

1. Install python 3.12
2. create the local python venv with `python3.12 -m venv venv`
3. Go inside the created venv `source venv/bin/activate`
4. Install all needed requirements with `pip install -r requirements.txt`
5. Make sure correct python interpreter is selected (select the venv one)
6. To locally run the server run `uvicorn app.main:app --reload --port 8000`
7. To go to swaggerUI go to `http://127.0.0.1:8000/docs`
