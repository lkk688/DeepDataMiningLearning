#https://fastapi.tiangolo.com/
#pip install fastapi #or pip install "fastapi[all]"
#pip install "uvicorn[standard]"
#pip install jinja2 #https://fastapi.tiangolo.com/advanced/templates/

#reference: https://christophergs.com/tutorials/ultimate-fastapi-tutorial-pt-6-jinja-templates/
#https://github.com/ChristopherGS/ultimate-fastapi-tutorial/blob/main/part-06-jinja-templates/app/templates/index.html

#uvicorn fastapitest:app --reload
#DeepDataMiningLearning\dataapps> uvicorn fastapitest:app --reload

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Union
from pydantic import BaseModel

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Any
from pathlib import Path

app = FastAPI(title="FastAPI test", openapi_url="/openapi.json") #The openapi_url parameter in FastAPI specifies the URL where the OpenAPI document for the API can be found
#app = FastAPI()
#API Document: http://127.0.0.1:8000/docs

#create a separate API router for your FastAPI application. This allows you to organize your API into different sections, each with its own set of routes.
#api_router = APIRouter()

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))
#templates = Jinja2Templates(directory="templates")

#The app.mount() method in FastAPI is used to mount a static file directory to the application. This allows you to serve static files, such as images, CSS, and JavaScript, from your application.
app.mount("/static", StaticFiles(directory=str(BASE_PATH / "static")), name="static")

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
@app.get("/itemsone/{item_id}")
def read_itemone(item_id: int, q: Union[str, None] = None):
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


#http://127.0.0.1:8000/items/3
@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return TEMPLATES.TemplateResponse(
        "item.html",
        {"request": request, "id": id},
    )

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    #http://127.0.0.1:8000/
