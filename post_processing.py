# Create a new dataframe with the index as numbers and final with column names index and predictions
import pickle
import re

import pandas as pd

with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# How many data points don't have Value: and Unit: in the prompt?
count = 0
for i in range(len(data)):
    if "Value:" not in data[i] or "Unit:" not in data[i]:
        print(i, data[i])
        data[i] = "Value: ''\nUnit: ''"

# Create two lists to store the values and units
values = []
units = []

# Extract the values and units from the data
for i in range(len(data)):
    value = data[i].split("Value: ")[1].split("\n")[0].strip()
    unit = data[i].split("Unit: ")[1].split("\n")[0].strip()
    values.append(value)
    units.append(unit)


# In case any of the values have units as well like any of the alphbets in it, split them and replace them in the unit
def split_values(arr):
    result = []
    for item in arr:
        match = re.match(r"(\d+\.?\d*)\s*([a-zA-Z]+)?", item)
        if match:
            num = match.group(1)
            unit = match.group(2) if match.group(2) else ""
            result.append((num, unit))
        else:
            result.append((item, ""))
    return result


values_temp = split_values(values)


for i in range(len(values_temp)):
    values[i] = values_temp[i][0]
    if values_temp[i][1] != "":
        units[i] = values_temp[i][1]

df = pd.read_csv("path_to_the_test_data")
entities = df["entity_name"].tolist()

# Iterate through the units and entities and correct the units
corrected_units = []

# Define mappings for each entity type
unit_mappings = {
    "width": {
        "f": "foot",
        "c": "centimetre",
        "i": "inch",
        "y": "yard",
        "m": ["metre", "millimetre"],
    },
    "depth": {
        "f": "foot",
        "c": "centimetre",
        "i": "inch",
        "y": "yard",
        "m": ["metre", "millimetre"],
    },
    "height": {
        "f": "foot",
        "c": "centimetre",
        "i": "inch",
        "y": "yard",
        "m": ["metre", "millimetre"],
    },
    "item_weight": {
        "l": "pound",
        "kg": "kilogram",
        "t": "ton",
        "oz": "ounce",
        "micro": "microgram",
        "mg": "milligram",
        "g": "gram",
    },
    "maximum_weight_recommendation": {
        "l": "pound",
        "kg": "kilogram",
        "t": "ton",
        "oz": "ounce",
        "micro": "microgram",
        "mg": "milligram",
        "g": "gram",
    },
    "voltage": {"k": "kilovolt", "m": "millivolt", "v": "volt"},
    "wattage": {"k": "kilowatt", "w": "watt"},
    "item_volume": {
        "oz": "fluid ounce",
        "fl": "fluid ounce",
        "cup": "cup",
        "q": "quart",
        "p": "pint",
        "g": "gallon",
        "ml": "millilitre",
        "ll": "millilitre",
        "f": "cubic foot",
        "in": "cubic inch",
        "i": "imperial gallon",
        "cl": "centilitre",
        "d": "decilitre",
        "m": "microlitre",
        "l": "litre",
    },
}


# Helper function to map unit based on the entity
def map_unit(unit, entity):
    unit = unit.strip().lower()
    if entity in unit_mappings:
        for key, value in unit_mappings[entity].items():
            if unit.startswith(key) or key in unit:
                # Handle cases where the value is a list (metre/millimetre)
                if isinstance(value, list):
                    return value[1] if unit.count("m") > 1 else value[0]
                return value
    return unit


# Main loop
corrected_units = [map_unit(unit, entity) for unit, entity in zip(units, entities)]

for i in range(len(corrected_units)):
    if corrected_units[i] not in [
        "centilitre",
        "centimetre",
        "cubic foot",
        "cubic inch",
        "cup",
        "decilitre",
        "fluid ounce",
        "foot",
        "gallon",
        "gram",
        "imperial gallon",
        "inch",
        "kilogram",
        "kilovolt",
        "kilowatt",
        "litre",
        "metre",
        "microgram",
        "microlitre",
        "milligram",
        "millilitre",
        "millimetre",
        "millivolt",
        "ounce",
        "pint",
        "pound",
        "quart",
        "ton",
        "volt",
        "watt",
        "yard",
    ]:
        corrected_units[i] = ""
        # df1["prediction"].iloc[i] = ""
        values[i] = ""

final = [(x + " " + y).strip() for x, y in zip(values, corrected_units)]

count = 0
for i in range(len(final)):
    if final[i] == "":
        count += 1

df2 = pd.read_csv("path_to_the_finetuned_csv_file")

df = pd.DataFrame(df2["index"].tolist(), columns=["index"])
df["prediction"] = final

df.to_csv("predictions.csv", index=False)
