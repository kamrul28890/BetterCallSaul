import xmltodict
import json
import pandas as pd
import random
# from dotenv import load_dotenv
# import os

# Warning control
import warnings
warnings.filterwarnings("ignore")

import json

transformed_data = []

# Open and read the .jsonl file line by line
with open('./data/text.data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Each line is a separate JSON object
        record = json.loads(line)
        
        # Extract the text from the first opinion in 'casebody_data' if it exists
        opinion_text = ""
        if 'casebody' in record and 'data' in record['casebody'] and record['casebody']['data']['opinions']:
            opinion_text = record['casebody']['data']['opinions'][0]['text']
        
        # Transform each record
        transformed_record = {
            'id': record['id'],
            'name': record['name'],
            'decision_date': record['decision_date'],
            'text': opinion_text,  # Place the extracted text here
            'source': 'https://case.law/'
        }
        
        # Add court name directly if it exists
        if 'court' in record:
            transformed_record['court'] = record['court'].get('name', None)
        
        # Append the transformed record to the list
        transformed_data.append(transformed_record)

# Save the transformed data to another JSON file
with open('./data/transformed.json', 'w', encoding='utf-8') as file:
    json.dump(transformed_data, file, indent=4, ensure_ascii=False)

print("Transformation complete. Data saved to 'transformed2.json'.")
