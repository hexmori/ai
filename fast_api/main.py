from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time

# ---------- 业务处理器 ----------
class ChatHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name
    async def handle(self, prompt: str):
        return f"[{self.model_name}] echo: {prompt}"

# ---------- 路由：从 state 取 handler ----------
class ChatRequest(BaseModel):
    prompt: str




@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 启动时 ----
    print("Starting up...")
    app.state.greeter = Greeter("world")  # 创建对象存到 state

    app.state.chat_handler = ChatHandler("my-model")   # ← 建 handler

    yield  # 服务在此对外提供

    # ---- 关闭时 ----
    print("Shutting down...")
    app.state.greeter.close()  # 释放资源


app = FastAPI(lifespan=lifespan)

# app.state  挂在 app 上的命名空间，用来存「整个应用共享的对象」
app.state.counter = 0
app.state.greeting = "hello"

@app.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    handler = request.app.state.chat_handler           # ← 取 handler
    result = await handler.handle(req.prompt)
    return {"result": result}


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []


class Greeter:
    def __init__(self, name: str):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

    def close(self):
        print("Greeter closed")


@app.post("/items")
def create_item(item: Item):
    return {"received": item.name, "total": item.price * 1.1, "tags": item.tags}


@app.get("/count")
def count():
    app.state.counter += 1
    return {"counter": app.state.counter, "greeting": app.state.greeting}


@app.get("/hello")
def hello(request: Request):
    # 从request中获取app.state
    g = request.app.state.greeting
    return {"message": g}


# ---------- 定义一个中间件函数 ----------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)      # 继续处理请求，拿到响应
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    print(f"{request.method} {request.url.path} took {duration:.4f}s")
    return response