import os
import pandas as pd

# ---------------- PATHS ----------------

pv_path = "dataset/plantVillage dataset/color"

synthetic_csv = "dataset/master_labels.csv"

# ---------------- LOAD SYNTHETIC ----------------

synthetic_df = pd.read_csv(
    synthetic_csv
)

synthetic_df = synthetic_df.fillna(0)

# ---------------- GET LABEL COLUMNS ----------------

label_cols = [

    c for c in synthetic_df.columns[1:]

    if c != "num_labels"
]

print(label_cols)

# ---------------- START RECORDS ----------------

records = []

# ---------------- ADD SYNTHETIC DATA ----------------

for _, row in synthetic_df.iterrows():

    row_values = []

    row_values.append(
        row["filename"]
    )

    for col in label_cols:

        row_values.append(
            row[col]
        )

    records.append(row_values)

# ---------------- FOLDER → LABEL MAPPING ----------------

folder_mapping = {

    "Apple___Apple_scab":
        "Apple___Apple_scab",

    "Apple___Black_rot":
        "Apple___Black_rot",

    "Apple___Cedar_apple_rust":
        "Apple___Cedar_apple_rust",

    "Tomato___Early_blight":
        "Tomato___Early_blight",

    "Tomato___Late_blight":
        "Tomato___Late_blight",

    "Tomato___Septoria_leaf_spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato___Target_Spot":
        "Tomato___Target_Spot",

    "Tomato___Tomato_mosaic_virus":
        "Tomato___Tomato_mosaic_virus",

    "Tomato___Tomato_YellowLeaf__Curl_Virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Grape___Black_rot":
        "Grape___Black_rot",

    "Grape___Esca_(Black_Measles)":
        "Grape___Esca_(Black_Measles)",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn_(maize)___Common_rust_":
        "Corn_(maize)___Common_rust_",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Corn_(maize)___Northern_Leaf_Blight",

    "Potato___Early_blight":
        "Potato___Early_blight",

    "Potato___Late_blight":
        "Potato___Late_blight"
}

# ---------------- ADD PLANTVILLAGE ----------------

for folder in os.listdir(pv_path):

    folder_path = os.path.join(
        pv_path,
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    if folder not in folder_mapping:
        continue

    label_name = folder_mapping[folder]

    files = os.listdir(folder_path)

    for file in files:

        if not file.lower().endswith(".jpg"):
            continue

        labels = [0]*len(label_cols)

        label_index = label_cols.index(
            label_name
        )

        labels[label_index] = 1

        row = [file] + labels

        records.append(row)

# ---------------- CREATE FINAL DF ----------------

columns = ["filename"] + label_cols

final_df = pd.DataFrame(
    records,
    columns=columns
)

# ---------------- REMOVE DUPLICATES ----------------

final_df = final_df.drop_duplicates(
    subset=["filename"]
)

# ---------------- SAVE ----------------

final_df.to_csv(
    "dataset/final_master_labels.csv",
    index=False
)

print(final_df.shape)

print("Combined CSV Created Successfully")