import dataclasses

@dataclasses.dataclass
class Address:
    input: list[str]
    output: list[str]



class Dog:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def do_something(self):
        self.weight = 100

if __name__ == "__main__":
    input_data = ["in1", "in2", "in3"]
    output_data = ["out1", "out2", "out3"]
    address = Address(input=input_data, output=output_data)
    print(address.input)
    (in1, in2, in3) = address.input  # list转元组
    print(in1)
    in_first, *_ = address.input  # list转元组
    print(in_first)

    # (in1, in2, in3) = [1, 2, 3]
    # print(in1)
    # int1, *_ = [1, 2, 3]
    # print(in1)
    # (int1,) = [1, 2]
    # print(int1)