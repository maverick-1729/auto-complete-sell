from fastapi import FastAPI
from pydantic import BaseModel
import autocomplete_trie

app = FastAPI()

class InputData(BaseModel):
    texts: list[str]
    query: str

@app.post("/autocomplete_sell/")
def predict(data: InputData):
    result = autocomplete_trie.get_suggestions_api(data.query, data.texts)  # Call your function
    return {"suggestions": result}
