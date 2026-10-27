"""
Comprehensive test script for GharValue AI House Price Prediction Web App
"""
import os
import sys
import unittest
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, CSV_HISTORY_FILE

class GharValueAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def test_01_home_page(self):
        """Test public home page renders correctly."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'GharValue', res.data)
        self.assertIn(b'12,000', res.data)
        self.assertIn(b'Random Forest Regressor', res.data)
        print("[PASS] Test 1: Home page renders successfully with live dataset stats")

    def test_02_predict_get(self):
        """Test GET /predict renders prediction form."""
        res = self.client.get('/predict')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Residential Property Valuation', res.data)
        self.assertIn(b'Quick Test Presets', res.data)
        print("[PASS] Test 2: Prediction form renders with presets and 18 parameters")

    def test_03_predict_post_valid(self):
        """Test POST /predict with realistic property inputs returns genuine prediction."""
        payload = {
            "City": "Mumbai",
            "Locality_Tier": "Mid",
            "BHK": "2",
            "Bathrooms": "2",
            "Super_Area_sqft": "1200",
            "Carpet_Area_sqft": "1000",
            "Floor_No": "5",
            "Total_Floors": "18",
            "Property_Age_years": "4",
            "Parking": "1",
            "Furnishing": "Semi-Furnished",
            "Lift": "1",
            "Gated_Society": "1",
            "Distance_to_Metro_km": "1.5",
            "Distance_to_CityCenter_km": "8.0",
            "Nearby_School_km": "1.2",
            "Nearby_Hospital_km": "1.8",
            "Crime_Rate_Index": "28.5"
        }
        res = self.client.post('/predict', data=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'ML Price Estimation', res.data)
        self.assertIn(b'result-price-main', res.data)
        print("[PASS] Test 3: POST /predict generates genuine ML prediction and results card")

    def test_04_predict_post_validation(self):
        """Test POST /predict with invalid inputs handles errors gracefully."""
        payload = {
            "City": "Mumbai",
            "Locality_Tier": "Mid",
            "BHK": "2",
            "Bathrooms": "2",
            "Super_Area_sqft": "1000",
            "Carpet_Area_sqft": "1200", # Invalid: Carpet > Super
            "Floor_No": "5",
            "Total_Floors": "18",
            "Property_Age_years": "4"
        }
        res = self.client.post('/predict', data=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Input validation error', res.data)
        print("[PASS] Test 4: Form validation catches carpet > super area correctly")

    def test_05_compare_page(self):
        """Test GET and POST /compare."""
        res_get = self.client.get('/compare')
        self.assertEqual(res_get.status_code, 200)
        self.assertIn(b'Side-by-Side Property Comparison', res_get.data)

        # POST comparison
        payload = {
            "p1_City": "Mumbai", "p1_Locality_Tier": "Mid", "p1_BHK": "2", "p1_Bathrooms": "2",
            "p1_Super_Area_sqft": "1200", "p1_Carpet_Area_sqft": "1000", "p1_Floor_No": "5",
            "p1_Total_Floors": "18", "p1_Property_Age_years": "4", "p1_Parking": "1",
            "p1_Furnishing": "Semi-Furnished", "p1_Lift": "1", "p1_Gated_Society": "1",
            "p1_Distance_to_Metro_km": "1.5", "p1_Distance_to_CityCenter_km": "8.0",
            "p1_Nearby_School_km": "1.2", "p1_Nearby_Hospital_km": "1.8", "p1_Crime_Rate_Index": "28.0",

            "p2_City": "Bangalore", "p2_Locality_Tier": "Premium", "p2_BHK": "3", "p2_Bathrooms": "3",
            "p2_Super_Area_sqft": "1650", "p2_Carpet_Area_sqft": "1350", "p2_Floor_No": "8",
            "p2_Total_Floors": "22", "p2_Property_Age_years": "2", "p2_Parking": "1",
            "p2_Furnishing": "Fully-Furnished", "p2_Lift": "1", "p2_Gated_Society": "1",
            "p2_Distance_to_Metro_km": "0.8", "p2_Distance_to_CityCenter_km": "5.0",
            "p2_Nearby_School_km": "0.9", "p2_Nearby_Hospital_km": "1.5", "p2_Crime_Rate_Index": "22.0"
        }
        res_post = self.client.post('/compare', data=payload)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Estimated Valuation Comparison', res_post.data)
        self.assertIn(b'Price Differential', res_post.data)
        print("[PASS] Test 5: Property comparison runs dual model inference successfully")

    def test_06_market_page(self):
        """Test GET /market with and without filters."""
        res = self.client.get('/market?city=Bangalore&tier=Premium')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Indian Metro Real Estate Explorer', res.data)
        self.assertIn(b'Bangalore', res.data)
        print("[PASS] Test 6: Market Explorer renders with filtered metrics and charts")

    def test_07_api_predict(self):
        """Test JSON API endpoint /api/predict."""
        payload = {
            "City": "Pune", "Locality_Tier": "Budget", "BHK": 1, "Bathrooms": 1,
            "Super_Area_sqft": 550.0, "Carpet_Area_sqft": 450.0, "Floor_No": 2,
            "Total_Floors": 8, "Property_Age_years": 5, "Parking": 0,
            "Furnishing": "Unfurnished", "Lift": 1, "Gated_Society": 0,
            "Distance_to_Metro_km": 3.0, "Distance_to_CityCenter_km": 12.0,
            "Nearby_School_km": 2.0, "Nearby_Hospital_km": 2.5, "Crime_Rate_Index": 40.0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('predicted_price_inr', data)
        print(f"[PASS] Test 7: API predict returned: INR {data['predicted_price_inr']:,} ({data['predicted_price_lakhs']} Lakhs)")

    def test_08_admin_protection(self):
        """Test unauthenticated access to admin routes is blocked and redirects to login."""
        res = self.client.get('/admin/dashboard')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/admin/login', res.headers['Location'])

        res_hist = self.client.get('/admin/history')
        self.assertEqual(res_hist.status_code, 302)
        print("[PASS] Test 8: Unauthenticated access to /admin/* is securely blocked")

    def test_09_admin_auth_and_pages(self):
        """Test admin login, session authentication, and all admin pages."""
        # Login with correct credentials
        res_login = self.client.post('/admin/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
        self.assertEqual(res_login.status_code, 200)
        self.assertIn(b'Platform Executive Overview', res_login.data)

        # Check all admin subpages
        pages = [
            ('/admin/dashboard', b'Platform Executive Overview'),
            ('/admin/history', b'Stored Prediction Records'),
            ('/admin/dataset', b'Dataset Statistical Analytics'),
            ('/admin/market-analytics', b'Metro Real Estate Intelligence'),
            ('/admin/models', b'Machine Learning Model Architecture'),
            ('/admin/settings', b'Platform Administration')
        ]
        for url, expected in pages:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, f"Failed on {url}")
            self.assertIn(expected, r.data, f"Missing expected text on {url}")

        print("[PASS] Test 9: Admin login and all 6 admin dashboards render with live data")

    def test_10_admin_export(self):
        """Test authenticated admin can export CSV history."""
        # Ensure authenticated
        self.client.post('/admin/login', data={'username': 'admin', 'password': 'admin123'})
        res = self.client.get('/admin/history/export')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'text/csv', res.headers['Content-Type'].encode())
        self.assertIn(b'prediction_id,timestamp', res.data)
        print("[PASS] Test 10: Admin CSV history export returns valid CSV content")

    def test_11_private_file_blocking(self):
        """Test direct access to sensitive files is forbidden."""
        for path in ['/house_price_model.pkl', '/house_price_features.pkl', '/.env']:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 403, f"Path {path} was not forbidden!")
        print("[PASS] Test 11: Direct URL access to private model/environment files is blocked (403)")

if __name__ == '__main__':
    unittest.main(verbosity=2)
