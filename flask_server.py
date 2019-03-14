from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def main():
	return render_template('index.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80, debug=True)
