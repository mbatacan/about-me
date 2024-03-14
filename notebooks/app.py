from flask import Flask, request, render_template
import src.query as query

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query_text = request.form['query']
        # Insert your LangChain query logic here
        results = query(query_text)
        return render_template('index.html', results=results)
    return render_template('index.html', results=[])


if __name__ == '__main__':
    app.run(debug=True)
