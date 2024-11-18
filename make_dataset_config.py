import json

import pandas as pd

entity_unit_map = {
    "width": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "depth": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "height": {"centimetre", "foot", "inch", "metre", "millimetre", "yard"},
    "item_weight": {
        "gram",
        "kilogram",
        "microgram",
        "milligram",
        "ounce",
        "pound",
        "ton",
    },
    "maximum_weight_recommendation": {
        "gram",
        "kilogram",
        "microgram",
        "milligram",
        "ounce",
        "pound",
        "ton",
    },
    "voltage": {"kilovolt", "millivolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {
        "centilitre",
        "cubic foot",
        "cubic inch",
        "cup",
        "decilitre",
        "fluid ounce",
        "gallon",
        "imperial gallon",
        "litre",
        "microlitre",
        "millilitre",
        "pint",
        "quart",
    },
}


train = pd.read_csv("path_to_the_train_csv_file")

data = []

for idx, row in train.iterrows():
    img = (
        "path_to_the_train_images_folder/"
        + row["image_link"].split("/")[-1]
    )
    instr = f"""<image>Extract the {row['entity_name']} of the item and its unit of measurement from the image. Ensure that the unit must be from the following: {entity_unit_map[row['entity_name']]}. Also make sure that if there is nothing to extract, you return ''. """
    ans = row["entity_value"]

    obj = {
        "messages": [
            {"content": instr, "role": "user"},
            {"content": ans, "role": "assistant"},
        ],
        "images": [img],
    }

    data.append(obj)

# save json
with open("train.json", "w") as f:
    json.dump(data, f)
