import os
import shutil

# ---------------- SOURCE ----------------

source_root = "dataset/plantVillage dataset/color"

# ---------------- DESTINATION ----------------

destination = "dataset/all_images"

os.makedirs(destination, exist_ok=True)

# ---------------- COPY ALL FOLDERS ----------------

total = 0

for folder in os.listdir(source_root):

    folder_path = os.path.join(
        source_root,
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    for file in files:

        if file.lower().endswith(".jpg"):

            src = os.path.join(
                folder_path,
                file
            )

            dst = os.path.join(
                destination,
                file
            )

            shutil.copy(src, dst)

            total += 1

print(f"Copied {total} images")