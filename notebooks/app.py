from flask import Flask, request, render_template, session
from flask_session import Session  # You might need to install this with pip

# Import your chatbot's query functionality here
from src.query import AboutMeBot
from src.db_connect import DBConnect
import src.fields as f
from src.etl import ETL

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize your chatbot here
client = DBConnect(f.PINECONE_API_KEY)  # type: ignore
index_name = "about-me"
index = client.pc.Index(index_name)
vdb = ETL('../data/resume.txt')
qa = AboutMeBot(vdb.vec_store)


@app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []  # Initialize an empty history

    if request.method == 'POST':
        user_query = request.form['query']  # Get the user's query from the form input
        response = qa.query(user_query)  # Query your chatbot

        # Append the user query and bot response to the history
        session['history'].append({'query': user_query, 'response': response['result']})
        session.modified = True  # To notify the session that we modified it

    return render_template('index.html', history=session['history'])


if __name__ == '__main__':
    app.run(debug=True)
