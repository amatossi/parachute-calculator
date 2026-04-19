import pickle
import json
import zipfile
import os
import numpy as np

repo_dir = r"C:\Users\arman\Documents\DoorDash\Parachute-Opening-Shock-main"
zip_path = os.path.join(repo_dir, "Pflanz Curve Data.zip")
pkl_path = os.path.join(repo_dir, "combined_data.pkl")

# Extract MIT coefficients from pickle
with open(pkl_path, "rb") as f:
    data = pickle.load(f)
mit_coeffs = data["MIT_coeffs"]

# Extract Pflanz Data from zip
pf_data = {}
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("pflanz_temp")

# The code in Data_Extraction.py joins Curve_1 and Curve_2 for each N
curve_cases = [5, 1, 2] # corresponding to n=0.5, 1, 2
dict_name = ['0.5', '1', '2']

for i in range(3):
    entire_curve = np.empty((0,2))
    for j in range(1, 3):
        file_name = f'X1_Curve_{j}_N{curve_cases[i]}.txt'
        file_path = os.path.join("pflanz_temp", file_name)
        data_array = np.loadtxt(file_path)
        if j > 1:
            data_array[:,0] *= 100
            data_array = np.vstack([data_array, [1000, 1]])
        entire_curve = np.vstack([entire_curve, data_array])
        
    pf_data[dict_name[i]] = {
        "x": entire_curve[:,0].tolist(),
        "y": entire_curve[:,1].tolist()
    }

# Export combined to json
out_data = {
    "MIT_coeffs": mit_coeffs,
    "Pflanz_knots": pf_data
}

with open(r"C:\Users\arman\.gemini\antigravity\scratch\parachute-calculator\data\shock_data.json", "w") as f:
    json.dump(out_data, f, indent=4)

print("Data successfully extracted to JSON.")

# Run the single value example to capture the output for validation
import sys
sys.path.append(repo_dir)
os.chdir(repo_dir)

try:
    with open("SingleValue_Example.py", "r") as f:
        code = f.read()
    # remove the plt.show() so it doesn't block
    code = code.replace("plt.show()", "")
    exec(code)
except Exception as e:
    print("Error running SingleValue_Example.py:", e)
