
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder=".")

# Load the trained model and feature names
model = joblib.load("house_price_model.pkl")
feature_names = joblib.load("house_price_features.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            house = {
                "City": request.form["City"],
                "Locality_Tier": request.form["Locality_Tier"],
                "BHK": int(request.form["BHK"]),
                "Bathrooms": int(request.form["Bathrooms"]),
                "Super_Area_sqft": float(request.form["Super_Area_sqft"]),
                "Carpet_Area_sqft": float(request.form["Carpet_Area_sqft"]),
                "Floor_No": int(request.form["Floor_No"]),
                "Total_Floors": int(request.form["Total_Floors"]),
                "Property_Age_years": int(request.form["Property_Age_years"]),
                "Parking": int(request.form["Parking"]),
                "Furnishing": request.form["Furnishing"],
                "Lift": int(request.form["Lift"]),
                "Gated_Society": int(request.form["Gated_Society"]),
                "Distance_to_Metro_km": float(request.form["Distance_to_Metro_km"]),
                "Distance_to_CityCenter_km": float(request.form["Distance_to_CityCenter_km"]),
                "Nearby_School_km": float(request.form["Nearby_School_km"]),
                "Nearby_Hospital_km": float(request.form["Nearby_Hospital_km"]),
                "Crime_Rate_Index": float(request.form["Crime_Rate_Index"])
            }

            input_df = pd.DataFrame([house], columns=feature_names)
            predicted_price = float(model.predict(input_df)[0])

            prediction = {
                "rupees": f"₹{predicted_price:,.0f}",
                "lakhs": f"₹{predicted_price / 100000:.2f} lakh"
            }

        except (ValueError, KeyError) as ex:
            error = f"Please check your inputs. Details: {ex}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
