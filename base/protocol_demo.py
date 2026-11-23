from typing import Protocol

class Quacker(Protocol):
    def quack(self) -> str:
        ...

class Duck:
    def quack(self) -> str:
        return "嘎嘎"

class Person:
    def quack(self) -> str:
        return "我在学鸭子叫"

    def show(self):
        pass

def make_it_quack(q: Quacker) -> None:
    print(q.quack())

make_it_quack(Duck())    # 嘎嘎
make_it_quack(Person())  # 我在学鸭子叫

# ========================================
class Named(Protocol):
    name: str          # 协议可以要求属性

class User:
    def __init__(self, name: str):
        self.name = name

class Config:
    name = "config"    # 类属性也算

def greet(x: Named) -> str:
    return f"Hello, {x.name}"

print(greet(User("Tom")))    # Hello, Tom
print(greet(Config()))       # Hello, config