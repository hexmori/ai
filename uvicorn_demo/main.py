from fast_api import FastAPI

app = FastAPI()


@app.get("/hello")
async def read_root():
    return {'message': 'xxxxxx'}


@app.get(path="/items/{item_id}")
async def get_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post(path="file_parse")
async def post_item(item_id: int, q: str):
    return {"item_id": item_id, "q": q}
