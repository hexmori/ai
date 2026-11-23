from typing import List, Dict, Callable, Any
import dataclasses
import msgspec


class User(msgspec.Struct):
    name: str
    groups: set[str] = set()
    email: str | None = None


@dataclasses.dataclass
class Student:
    name: str
    age: int


def wwwwxxx(call: Callable[[Any], None]):
    call(100)


student = Student("Tom", 22)
fields = dataclasses.fields(type(student))
print(fields)
key_value_dataclass_dict: dict = {}
for field in fields:
    key_value_dataclass_dict[field.name] = getattr(student, field.name)
print(key_value_dataclass_dict)

user = User("Tom", [1, 2], "xxx@xx.com")
fields = msgspec.structs.fields(user)
key_value_msgspec_dict: dict = {}
for field in fields:
    key_value_msgspec_dict[field.name] = getattr(user, field.name)
print(key_value_msgspec_dict)

_HOOKS: Dict[str, List[Callable[[Any, Callable[[Any], None]], None]]] = {}

list1: List[Callable[[Any, Callable[[Any], None]], None]] = []
call1: Callable[[Any, Callable[[Any], None]], None] = None
