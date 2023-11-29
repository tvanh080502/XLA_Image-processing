from flask import Flask, redirect, render_template
from ImageProcessing import IP

app = Flask(__name__)
app.config["SECRET_KEY"] = "ddki1912"
app.register_blueprint(IP)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
