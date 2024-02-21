#https://fastapi.tiangolo.com/
#pip install fastapi #or pip install "fastapi[all]"
#pip install "uvicorn[standard]"
#pip install jinja2 #https://fastapi.tiangolo.com/advanced/templates/

#uvicorn fastapitest:app --reload
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#API Document: http://127.0.0.1:8000/docs

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

#http://127.0.0.1:8000/
@app.get("/")
def read_root():
    return {"Hello": "World"}


#http://127.0.0.1:8000/items/3?q=test
#The path /items/{item_id} has a path parameter item_id that should be an int.
#The path /items/{item_id} has an optional str query parameter q
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

r"""
curl -X 'PUT' \
  'http://127.0.0.1:8000/items/3' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "ttt",
  "price": 0,
  "is_offer": true
}'
"""
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}