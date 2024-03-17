from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
import re
import uvicorn
import requests
import logging

# Set logging level and format
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize LangChain and LLMChain
llm_restro = OpenAI(temperature=0.5)
prompt_template = PromptTemplate(
    input_variables=['age', 'gender', 'symptoms'],
    template="Drug Recommendation System:\n"
             "I want you to recommend 4 medicines and 4 precautions for a disease "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person symptoms: {symptoms}\n"
)
chain_restro = LLMChain(llm=llm_restro, prompt=prompt_template)

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Define endpoint to recommend medicines and precautions
@app.post("/recommendations/")
def get_recommendations(age: int, gender: str, symptoms: str):
    input_data = {
        'age': age,
        'gender': gender,
        'symptoms': symptoms
    }
    try:
        logging.info(f"Received request: age={age}, gender={gender}, symptoms={symptoms}")
        
        # Generate recommendations using LangChain
        result = chain_restro.run(input_data)
        
        logging.info(f"Generated recommendations: {result}")
        
        medicine = re.findall(r"Medicine Recommendations:(.*?)\n\n", result, re.DOTALL)
        precaution = re.findall(r'Precautions:(.*?)$', result, re.DOTALL)

        medicine_names = [name.strip() for name in medicine[0].strip().split('\n') if name.strip()] if medicine else []
        precautions = [name.strip() for name in precaution[0].strip().split('\n') if name.strip()] if precaution else []
        
        return {"medicine_recommendations": medicine_names, "precautions": precautions}
    except Exception as e:
        # Handle exceptions
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server using Uvicorn
if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)


