from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_session import Session
from src.query import AboutMeBot
from src.db_connect import DBConnect
import src.fields as f
from src.etl import ETL
import wandb


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config['SECRET_KEY'] = f.SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

client = DBConnect(f.PINECONE_API_KEY)  # type: ignore
index_name = "about-me"
index = client.pc.Index(index_name)

# todo implement adding new text through website
vdb = ETL('../data/about_me.txt')

# Initialize chatbot here
qa = AboutMeBot(vdb.vec_store)


@app.route('/reset')
def reset_session():
    session.clear()
    return redirect(url_for('home'))


@app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []  # Initialize an empty history

    if request.method == 'POST':
        user_query = request.form['query']  # Get the user's query from the form input
        response = qa.query(user_query)  # Query your chatbot
        wandb.log(
            {"user_message": user_query, "bot_response": response['result']}
        )  # Fix: Access 'result' using square bracket notation
        # Append the user query and bot response to the history
        session['history'].append({'query': user_query, 'response': response['result']})  # type: ignore
        session.modified = True  # To notify the session that we modified it

    return render_template('index.html', history=session['history'])


@app.route('/upload-text', methods=['POST'])
def upload_text():
    new_text = request.form['new_text']
    if new_text.strip():
        qa._add_new_docs(new_text)
        flash('New text has been added successfully.')
    else:
        flash('Please submit non-empty text.')

    return redirect(url_for('home'))


if __name__ == '__main__':
    wandb.init(project='about-me-chatbot', entity='flask-chatbot')
    app.run(debug=True)
