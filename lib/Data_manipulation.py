import json
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def split_form10k_data_from_file(json):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
    )
    chunks_with_metadata = [] # use this to accumlate chunk records
    # file_as_object = json.load(open(file)) # open the json file
    file_as_object = json
    # for item in ['item1','item1a','item7','item7a']: # pull these keys from the json
    
    # print(f'Processing {item} from {file}') 
    item_text = file_as_object['text'] # grab the text of the item
    item_text_chunks = text_splitter.split_text(item_text) # split the text into chunks
    chunk_seq_id = 0
    for chunk in item_text_chunks[:20]: # only take the first 20 chunks
        # form_id = file[file.rindex('/') + 1:file.rindex('.')] # extract form id from file name
        # finally, construct a record with metadata and the chunk text
        chunks_with_metadata.append({
            'text': chunk, 
            'id':file_as_object['id'],
            'chunkId': f'{file_as_object['id']}-chunk{chunk_seq_id:04d}',
            'chunkSeqId': chunk_seq_id,
            'name': file_as_object['name'],
            'decision_date': file_as_object['decision_date'],
            'source': file_as_object['source'],
            'court': file_as_object['court']
        })
        chunk_seq_id += 1
    return chunks_with_metadata


def return_chunk_data():

    with open('./data/transformed.json', 'r', encoding='utf-8') as file:
        transformed_data = json.load(file)

    random.seed(42)
    transformed_data = random.sample(transformed_data, 100)
    # print(len(transformed_data))

    first_case = []
    for i in range(0, len(transformed_data)):
        case = transformed_data[i]
        first_case.extend(split_form10k_data_from_file(case))
    
    # Initialize the OpenAI client with your API key
    OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
    client = OpenAI(api_key=OPEN_AI_API_KEY)

    # Sample code to iterate through each case and classify
    for case in first_case:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "I have the description of a case, classify the case into any one of the following: [Murder, Theft, Arson, Drug_Trafficking, Rape, Sexual_Assault, Child_Pornography, Fraud, Embezzlment, Money_Laundring, Insider_Trading, Tax_Evasion, Public_Intoxication, Disorderly_Conduct, Hacking, Identity_Theft, Driving_Under_Influence, Speeding, Real_Estate_Dispute, Family_law]. Choose only from the items I provided. If you cannot match the content based on the list, reply N/A. Also return only the key word as case type, do not make it a sentance. The case type should be included in the list I gave"
                    },
                    {
                        "role": "user",
                        "content": case['text']  # Use the 'text' from the current case in the loop
                    }
                ],
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            case_type = response.choices[0].message.content  # Hypothetical parsing of the response
            case['case_type'] = case_type  # Add the case_type to the case dictionary

        except Exception as e:
            print(f"An error occurred: {e}")
            case['case_type'] = 'Error processing case'  # Handle errors gracefully

    for item in first_case:
        item['text'] = f"The name of the case is {item['name']}. The decision date is ({item['decision_date']})\n\n{item['text']}"

    return first_case

def return_transformed_data():
    with open('./data/transformed.json', 'r', encoding='utf-8') as file:
        transformed_data = json.load(file)

    random.seed(42)
    transformed_data = random.sample(transformed_data, 100)

    return transformed_data


