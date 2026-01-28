class StringManipulator:
    def __init__(self):
        self.s = ""

    def get_String(self):
        self.s = input("Enter a string: ")

    def print_String(self):
        print("String in uppercase:", self.s.upper())

obj = StringManipulator()
obj.get_String()
obj.print_String()
