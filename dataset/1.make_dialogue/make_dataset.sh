#!/bin/bash

MODEL_NAME='gpt-4o'
API_KEY=''
PERSONA_PATH='./Processed_PersonaChat.json'
RELATIONSHIP_PATH='./relationships_list.txt'
DATASET_COUNT=10
OUPTUT_PATH='../data.json'

python generate_dialogue.py --model_name $MODEL_NAME --api_key $API_KEY --number_of_datasets $DATASET_COUNT --persona_path $PERSONA_PATH --relationship_path $RELATIONSHIP_PATH --output_path $OUPTUT_PATH
