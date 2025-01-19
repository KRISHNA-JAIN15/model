from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)


app = Flask(__name__)


species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)
            predicted_species = species_mapping[prediction[0]]

            return render_template("predict.html", species=predicted_species)
        except ValueError:
            return "Invalid input. Please enter numeric values for all fields."

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        
        input_data = np.array([[data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]])
        prediction = model.predict(input_data)
        predicted_species = species_mapping[prediction[0]]
        return jsonify({"predicted_species": predicted_species})
    except KeyError:
        return jsonify({"error": "Invalid input format. Ensure all features are provided."}), 400

if __name__ == "__main__":
    app.run(debug=True)
