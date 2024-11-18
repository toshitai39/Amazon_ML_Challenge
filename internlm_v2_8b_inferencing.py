import pickle
import pandas as pd
import nest_asyncio
from lmdeploy import pipeline, TurbomindEngineConfig
from tqdm import tqdm

nest_asyncio.apply()


test = pd.read_csv("./dataset/test.csv")
# Prompt
PROMPT = """Extract only one item and its unit of measurement from the image in the format -
Value: <value> \nUnit: <unit of measurement>
In case of no value or unit of measurement found, return 'Value: ' \nUnit: '."""
prompt = PROMPT.split("item")
prompts = [
    prompt[0] + "item " + test["entity_name"].iloc[idx] + prompt[-1]
    for idx in range(len(test))
]

test["index"] = test["image_link"].apply(lambda x: x.split("/")[-1])
images = [f"./train/{test['index'].iloc[idx]}" for idx in range(len(test))]

# Model for inferencing
model = "OpenGVLab/InternVL2-8B"
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=None))

# Save the responses to a csv file
responses = []
count = 0
for i in tqdm(range(0, len(prompts), 32)):
    prom = [(prompts[i + j], images[i + j]) for j in range(min(32, len(prompts) - i))]
    response = pipe(prom)
    for j in range(min(32, len(prompts) - i)):
        responses.append(response[j].text)
    count += 1
    if count % 20 == 0:
        with open("data.pkl", "wb") as f:
            pickle.dump(responses, f)

with open("data.pkl", "wb") as f:
    pickle.dump(responses, f)
