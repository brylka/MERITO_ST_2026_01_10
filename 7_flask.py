from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return "<style>* {background-color: red;}</style><h1>Witaj świecie!</h1>"

if __name__ == '__main__':
    app.run(debug=True)