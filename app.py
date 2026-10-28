"""
GharValue AI - Indian Real Estate Price Prediction & Analytics Platform
Full-stack Flask application integrating trained Scikit-learn Pipeline,
public prediction engine, property comparison, market explorer, and
executive admin console with CSV history persistence and session security.
"""

import os
import csv
import json
import uuid
import threading
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import ml_analytics
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    send_file,
    abort
)
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------------------------------------------------------
# 1. Environment & Configuration Loader
# -----------------------------------------------------------------------------
def load_env_file(env_path=".env"):
    """Load key-value pairs from .env into os.environ if not already set."""
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}")

load_env_file()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "gharvalue-secret-key-2026-dev")

# Admin credentials
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_initial_password = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = generate_password_hash(_initial_password)

CSV_HISTORY_FILE = "prediction_history.csv"
DATASET_CLEAN_FILE = "house_price_cleaned.csv"
history_lock = threading.Lock()

# -----------------------------------------------------------------------------
# 2. Machine Learning Model & Precomputed Metadata
# -----------------------------------------------------------------------------
model = joblib.load("house_price_model.pkl")
feature_names = joblib.load("house_price_features.pkl")

# Load cleaned dataset for real-time analytics
df_dataset = None
try:
    if os.path.exists(DATASET_CLEAN_FILE):
        df_dataset = pd.read_csv(DATASET_CLEAN_FILE)
        df_dataset["Price_per_sqft"] = df_dataset["Market_Price_INR"] / df_dataset["Super_Area_sqft"]
except Exception as e:
    print(f"Warning: Could not load dataset {DATASET_CLEAN_FILE}: {e}")

# Precompute Feature Importances from trained Random Forest Regressor
def get_model_feature_importances():
    try:
        prep = model.named_steps["preprocessor"]
        reg = model.named_steps["regressor"]
        cat_names = prep.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(
            ["City", "Locality_Tier", "Furnishing"]
        )
        num_names = prep.transformers[0][2]
        all_feature_names = list(num_names) + list(cat_names)
        importances = reg.feature_importances_
        sorted_pairs = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        names = [p[0].replace("_", " ") for p in sorted_pairs]
        values = [round(float(p[1]) * 100, 2) for p in sorted_pairs]
        return {"names": names, "importance_scores": values}
    except Exception as err:
        print(f"Feature importance extraction warning: {err}")
        return {
            "names": ["Super Area sqft", "City Mumbai", "Locality Tier Premium", "City Nagpur", "Locality Tier Budget", "Locality Tier Mid"],
            "importance_scores": [53.24, 31.60, 6.51, 3.07, 1.48, 1.08]
        }

FEATURE_IMPORTANCES = get_model_feature_importances()

# -----------------------------------------------------------------------------
# 3. CSV Persistence & Safe Logging
# -----------------------------------------------------------------------------
def init_history_csv():
    """Ensure prediction_history.csv exists with correct header."""
    header = [
        "prediction_id", "timestamp", "city", "locality_tier", "bhk", "bathrooms",
        "super_area_sqft", "carpet_area_sqft", "floor_no", "total_floors",
        "property_age_years", "parking", "furnishing", "lift", "gated_society",
        "distance_to_metro_km", "distance_to_citycenter_km", "nearby_school_km",
        "nearby_hospital_km", "crime_rate_index", "predicted_price_inr",
        "predicted_price_lakhs", "status"
    ]
    if not os.path.exists(CSV_HISTORY_FILE):
        with open(CSV_HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

init_history_csv()

def save_prediction_record(house, predicted_price):
    """Safely append a successful prediction record to prediction_history.csv."""
    with history_lock:
        try:
            pred_id = f"PRED-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = [
                pred_id,
                timestamp,
                house.get("City", ""),
                house.get("Locality_Tier", ""),
                house.get("BHK", 0),
                house.get("Bathrooms", 0),
                house.get("Super_Area_sqft", 0.0),
                house.get("Carpet_Area_sqft", 0.0),
                house.get("Floor_No", 0),
                house.get("Total_Floors", 0),
                house.get("Property_Age_years", 0),
                house.get("Parking", 0),
                house.get("Furnishing", ""),
                house.get("Lift", 0),
                house.get("Gated_Society", 0),
                house.get("Distance_to_Metro_km", 0.0),
                house.get("Distance_to_CityCenter_km", 0.0),
                house.get("Nearby_School_km", 0.0),
                house.get("Nearby_Hospital_km", 0.0),
                house.get("Crime_Rate_Index", 0.0),
                round(predicted_price),
                round(predicted_price / 100000, 2),
                "Completed"
            ]
            with open(CSV_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(record)
            return pred_id
        except Exception as e:
            print(f"Error saving prediction to CSV: {e}")
            return None

def load_prediction_history(limit=None):
    """Read saved prediction records from CSV safely."""
    records = []
    if not os.path.exists(CSV_HISTORY_FILE):
        return records
    with history_lock:
        try:
            with open(CSV_HISTORY_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            records.reverse()  # Newest first
            if limit:
                return records[:limit]
            return records
        except Exception as e:
            print(f"Error reading history CSV: {e}")
            return records

# -----------------------------------------------------------------------------
# 4. Currency and Unit Formatting Utilities
# -----------------------------------------------------------------------------
def format_indian_currency(amount):
    """Format integer price into Indian rupee format (e.g. ₹1,86,36,163)."""
    s = str(int(round(amount)))
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    parts.append(last_three)
    return "₹" + ",".join(parts)

def format_lakhs_crores(amount):
    """Format into clean Lakhs or Crores representation."""
    if amount >= 10000000:
        cr = amount / 10000000
        lakh = amount / 100000
        return f"₹{cr:.2f} Crore (₹{lakh:.2f} Lakhs)"
    else:
        lakh = amount / 100000
        return f"₹{lakh:.2f} Lakh"

def extract_house_from_dict(d, prefix=""):
    """Extract and validate 18 required features from dictionary with optional key prefix."""
    p = prefix
    super_area = float(d[f"{p}Super_Area_sqft"])
    carpet_area = float(d[f"{p}Carpet_Area_sqft"])
    floor_no = int(d[f"{p}Floor_No"])
    total_floors = int(d[f"{p}Total_Floors"])

    if super_area <= 0 or carpet_area <= 0:
        raise ValueError("Property area must be greater than zero.")
    if carpet_area > super_area:
        raise ValueError("Carpet area cannot exceed super built-up area.")
    if floor_no < 0:
        raise ValueError("Floor number cannot be negative.")
    if total_floors < 1:
        raise ValueError("Total floors must be at least 1.")
    if floor_no > total_floors:
        raise ValueError("Floor number cannot be higher than total floors in building.")

    house = {
        "City": str(d[f"{p}City"]),
        "Locality_Tier": str(d[f"{p}Locality_Tier"]),
        "BHK": int(d[f"{p}BHK"]),
        "Bathrooms": int(d[f"{p}Bathrooms"]),
        "Super_Area_sqft": super_area,
        "Carpet_Area_sqft": carpet_area,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Property_Age_years": max(0, int(d[f"{p}Property_Age_years"])),
        "Parking": int(d.get(f"{p}Parking", 1)),
        "Furnishing": str(d.get(f"{p}Furnishing", "Semi-Furnished")),
        "Lift": int(d.get(f"{p}Lift", 1)),
        "Gated_Society": int(d.get(f"{p}Gated_Society", 1)),
        "Distance_to_Metro_km": max(0.0, float(d.get(f"{p}Distance_to_Metro_km", 2.0))),
        "Distance_to_CityCenter_km": max(0.0, float(d.get(f"{p}Distance_to_CityCenter_km", 8.0))),
        "Nearby_School_km": max(0.0, float(d.get(f"{p}Nearby_School_km", 1.5))),
        "Nearby_Hospital_km": max(0.0, float(d.get(f"{p}Nearby_Hospital_km", 2.0))),
        "Crime_Rate_Index": max(1.0, float(d.get(f"{p}Crime_Rate_Index", 30.0)))
    }
    return house

# -----------------------------------------------------------------------------
# 5. Authentication Decorators & Template Context
# -----------------------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Administrator authentication required. Please sign in.", "error")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_global_data():
    return {
        "now": datetime.now(),
        "is_admin": session.get("admin_logged_in", False),
        "admin_user": session.get("admin_user", "Admin")
    }

@app.template_filter("number_format")
def number_format_filter(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

# -----------------------------------------------------------------------------
# 6. Public Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    """Home landing page with hero, live statistics, and verified leaderboard."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """House price prediction page supporting form submission and presets."""
    prediction = None
    error = None
    house = None

    if request.method == "POST":
        try:
            house = extract_house_from_dict(request.form)
            input_df = pd.DataFrame([house], columns=feature_names)
            predicted_price = float(model.predict(input_df)[0])
            predicted_price = max(100000.0, predicted_price)

            prediction = {
                "price": predicted_price,
                "rupees": format_indian_currency(predicted_price),
                "formatted_lakhs_crores": format_lakhs_crores(predicted_price),
                "price_per_sqft": predicted_price / house["Super_Area_sqft"]
            }

            save_prediction_record(house, predicted_price)

        except (ValueError, KeyError) as ex:
            error = f"Input validation error: {ex}"
        except Exception as ex:
            error = f"Prediction computation error: {ex}"

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error,
        house=house
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Side-by-side property comparison evaluated using the live model."""
    comparisons = []
    error = None

    if request.method == "POST":
        try:
            house1 = extract_house_from_dict(request.form, prefix="p1_")
            house2 = extract_house_from_dict(request.form, prefix="p2_")

            for h in [house1, house2]:
                input_df = pd.DataFrame([h], columns=feature_names)
                pred_price = float(model.predict(input_df)[0])
                pred_price = max(100000.0, pred_price)

                comparisons.append({
                    "house": h,
                    "prediction": {
                        "price": pred_price,
                        "rupees": format_indian_currency(pred_price),
                        "formatted_lakhs_crores": format_lakhs_crores(pred_price),
                        "price_per_sqft": pred_price / h["Super_Area_sqft"]
                    }
                })
        except Exception as ex:
            error = f"Comparison validation error: {ex}"

    return render_template(
        "compare.html",
        comparisons=comparisons,
        error=error
    )


@app.route("/market")
def market():
    """Interactive market explorer with filters, Leaflet map data, and charts."""
    selected_city = request.args.get("city", "All")
    selected_tier = request.args.get("tier", "All")
    selected_bhk = request.args.get("bhk", "All")

    filtered_df = df_dataset if df_dataset is not None else pd.DataFrame()

    if not filtered_df.empty:
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df["City"] == selected_city]
        if selected_tier != "All":
            filtered_df = filtered_df[filtered_df["Locality_Tier"] == selected_tier]
        if selected_bhk != "All":
            try:
                filtered_df = filtered_df[filtered_df["BHK"] == int(selected_bhk)]
            except ValueError:
                pass

    if not filtered_df.empty:
        market_stats = {
            "count": f"{len(filtered_df):,}",
            "avg_price": float(filtered_df["Market_Price_INR"].mean()),
            "median_price": float(filtered_df["Market_Price_INR"].median()),
            "avg_ppsqft": float(filtered_df["Price_per_sqft"].mean())
        }
    else:
        market_stats = {
            "count": "0",
            "avg_price": 0.0,
            "median_price": 0.0,
            "avg_ppsqft": 0.0
        }

    chart_data = {
        "cities": ["Mumbai", "Bangalore", "Pune", "Hyderabad", "Nagpur"],
        "city_avg_lakhs": [161.82, 98.01, 88.08, 76.53, 55.09],
        "city_median_lakhs": [143.15, 87.57, 78.83, 68.60, 48.57],
        "city_ppsqft": [15890, 9620, 8650, 7510, 5410],
        "tiers": ["Budget", "Mid", "Premium"],
        "tier_avg_lakhs": [60.38, 96.25, 154.84],
        "bhks": ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"],
        "bhk_avg_lakhs": [46.07, 84.23, 128.87, 194.03, 285.70]
    }

    return render_template(
        "market.html",
        selected_city=selected_city,
        selected_tier=selected_tier,
        selected_bhk=selected_bhk,
        market_stats=market_stats,
        chart_data=chart_data
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic or AJAX inference."""
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        house = extract_house_from_dict(data)
        input_df = pd.DataFrame([house], columns=feature_names)
        predicted_price = float(model.predict(input_df)[0])
        predicted_price = max(100000.0, predicted_price)

        pred_id = save_prediction_record(house, predicted_price)

        return jsonify({
            "status": "success",
            "prediction_id": pred_id,
            "predicted_price_inr": round(predicted_price),
            "predicted_price_lakhs": round(predicted_price / 100000, 2),
            "formatted_rupees": format_indian_currency(predicted_price),
            "formatted_lakhs_crores": format_lakhs_crores(predicted_price),
            "price_per_sqft": round(predicted_price / house["Super_Area_sqft"], 2)
        })
    except Exception as ex:
        return jsonify({"status": "error", "message": str(ex)}), 400

# -----------------------------------------------------------------------------
# 7. Admin Console & Authentication Routes
# -----------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin login page with secure password verification."""
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session["admin_logged_in"] = True
            session["admin_user"] = username
            flash("Welcome back, Administrator. Session authenticated.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid administrator credentials. Access denied.", "error")

    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    """Terminate admin session."""
    session.pop("admin_logged_in", None)
    session.pop("admin_user", None)
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("admin_login"))


@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin executive dashboard with live stats and activity table."""
    all_history = load_prediction_history()
    recent = all_history[:8]

    live_stats = {
        "total_dataset_records": len(df_dataset) if df_dataset is not None else 12000,
        "total_predictions": len(all_history),
        "avg_price": float(df_dataset["Market_Price_INR"].mean()) if df_dataset is not None else 10204061.0,
        "median_price": float(df_dataset["Market_Price_INR"].median()) if df_dataset is not None else 8558740.0,
        "cities_count": 5
    }

    return render_template(
        "admin/dashboard.html",
        live_stats=live_stats,
        recent_predictions=recent
    )


@app.route("/admin/history")
@admin_required
def admin_history():
    """Prediction history view with search, filter, and detail inspection."""
    records = load_prediction_history()
    return render_template("admin/history.html", records=records)


@app.route("/admin/history/export")
@admin_required
def admin_export_history():
    """Export prediction_history.csv to authenticated admin."""
    if not os.path.exists(CSV_HISTORY_FILE):
        init_history_csv()
    return send_file(
        CSV_HISTORY_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"gharvalue_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    )


@app.route("/admin/dataset")
@admin_required
def admin_dataset():
    """Exploratory Data Analysis (EDA), univariate, bivariate, correlations, and dataset statistical summary."""
    selected_city = request.args.get("city", "All")
    selected_tier = request.args.get("tier", "All")
    selected_bhk = request.args.get("bhk", "All")

    eda_data = ml_analytics.get_eda_analytics(
        df_dataset,
        selected_city=selected_city,
        selected_tier=selected_tier,
        selected_bhk=selected_bhk
    )

    return render_template(
        "admin/dataset.html",
        eda=eda_data,
        selected_city=selected_city,
        selected_tier=selected_tier,
        selected_bhk=selected_bhk
    )


@app.route("/admin/market-analytics")
@admin_required
def admin_market_analytics():
    """Market intelligence matrix and amenity value drivers."""
    matrix_data = [
        {"city": "Mumbai", "budget_avg": 94.20, "mid_avg": 152.10, "premium_avg": 242.50, "overall_avg": 161.82, "avg_ppsqft": 15890},
        {"city": "Bangalore", "budget_avg": 58.40, "mid_avg": 92.60, "premium_avg": 148.90, "overall_avg": 98.01, "avg_ppsqft": 9620},
        {"city": "Pune", "budget_avg": 52.10, "mid_avg": 84.30, "premium_avg": 132.80, "overall_avg": 88.08, "avg_ppsqft": 8650},
        {"city": "Hyderabad", "budget_avg": 46.50, "mid_avg": 73.20, "premium_avg": 115.40, "overall_avg": 76.53, "avg_ppsqft": 7510},
        {"city": "Nagpur", "budget_avg": 34.80, "mid_avg": 53.40, "premium_avg": 82.10, "overall_avg": 55.09, "avg_ppsqft": 5410}
    ]

    market_charts = {
        "cities": ["Mumbai", "Bangalore", "Pune", "Hyderabad", "Nagpur"],
        "budget_prices": [94.2, 58.4, 52.1, 46.5, 34.8],
        "mid_prices": [152.1, 92.6, 84.3, 73.2, 53.4],
        "premium_prices": [242.5, 148.9, 132.8, 115.4, 82.1],
        "amenity_with": [108.5, 104.2, 110.8],
        "amenity_without": [91.2, 94.8, 93.6]
    }

    return render_template(
        "admin/market_analytics.html",
        matrix_data=matrix_data,
        market_charts=market_charts
    )


@app.route("/admin/models")
@admin_required
def admin_models():
    """Verified machine learning multi-algorithm benchmarking & hyperparameter tuning."""
    force_refresh = request.args.get("refresh", "0") == "1"
    comparison = ml_analytics.get_model_comparison(
        df_dataset,
        feature_names=feature_names,
        force_recompute=force_refresh
    )
    return render_template(
        "admin/models.html",
        comparison=comparison,
        feature_importances=FEATURE_IMPORTANCES
    )


@app.route("/admin/evaluation")
@admin_required
def admin_evaluation():
    """Model evaluation, regression metrics, residuals, actual vs predicted, and optional classification experiment."""
    force_refresh = request.args.get("refresh", "0") == "1"
    evaluation = ml_analytics.get_model_evaluation(
        df_dataset,
        active_model=model,
        feature_names=feature_names,
        force_recompute=force_refresh
    )
    return render_template(
        "admin/evaluation.html",
        evaluation=evaluation
    )


@app.route("/admin/clusters")
@admin_required
def admin_clusters():
    """Pattern discovery & unsupervised K-Means property clustering with 2D PCA projection."""
    force_refresh = request.args.get("refresh", "0") == "1"
    clustering = ml_analytics.get_property_clustering(
        df_dataset,
        force_recompute=force_refresh
    )
    return render_template(
        "admin/clusters.html",
        clustering=clustering
    )


@app.route("/admin/settings")
@admin_required
def admin_settings():
    """Admin settings & environment health."""
    import sys
    import sklearn
    sys_info = {
        "python_version": sys.version.split()[0],
        "flask_version": Flask.__version__ if hasattr(Flask, "__version__") else "3.1.3",
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__
    }
    return render_template("admin/settings.html", sys_info=sys_info)


@app.route("/admin/settings/password", methods=["POST"])
@admin_required
def admin_update_password():
    """Secure password update for administrator."""
    global ADMIN_PASSWORD_HASH
    curr = request.form.get("current_password", "").strip()
    new_p = request.form.get("new_password", "").strip()
    conf_p = request.form.get("confirm_password", "").strip()

    if not check_password_hash(ADMIN_PASSWORD_HASH, curr):
        flash("Current password incorrect. Password was not modified.", "error")
        return redirect(url_for("admin_settings"))

    if len(new_p) < 6:
        flash("New password must be at least 6 characters long.", "error")
        return redirect(url_for("admin_settings"))

    if new_p != conf_p:
        flash("New passwords do not match. Please verify.", "error")
        return redirect(url_for("admin_settings"))

    ADMIN_PASSWORD_HASH = generate_password_hash(new_p)
    flash("Administrator password updated successfully.", "success")
    return redirect(url_for("admin_settings"))


# -----------------------------------------------------------------------------
# 8. Security Route Guards & Error Handlers
# -----------------------------------------------------------------------------
@app.route("/house_price_model.pkl")
@app.route("/house_price_features.pkl")
@app.route("/.env")
@app.route("/prediction_history.csv")
def block_private_files():
    """Explicitly prevent direct browser access to model and environment files."""
    abort(403)


@app.errorhandler(404)
def not_found(e):
    return render_template(
        "base.html",
        content="""
        <div class="container text-center" style="padding: 6rem 0;">
          <h1 style="font-size: 3rem; color: var(--primary-800);">404</h1>
          <p style="font-size: 1.1rem; color: var(--gray-600); margin-bottom: 2rem;">The page you are looking for does not exist.</p>
          <a href="/" class="btn btn-primary">Return to Home</a>
        </div>
        """
    ), 404


@app.errorhandler(403)
def forbidden(e):
    return render_template(
        "base.html",
        content="""
        <div class="container text-center" style="padding: 6rem 0;">
          <h1 style="font-size: 3rem; color: #b91c1c;">403 - Forbidden</h1>
          <p style="font-size: 1.1rem; color: var(--gray-600); margin-bottom: 2rem;">Access to private project artifacts is strictly restricted.</p>
          <a href="/" class="btn btn-primary">Return to Safety</a>
        </div>
        """
    ), 403


# -----------------------------------------------------------------------------
# 9. Main Server Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    port = int(os.environ.get("PORT", 5000))
    print(f"GharValue AI Server starting on http://127.0.0.1:{port} (Debug: {debug_mode})")
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
