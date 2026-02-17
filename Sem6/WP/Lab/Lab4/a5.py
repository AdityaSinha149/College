class ReverseWords:
    def reverse_words(self, s):
        words = s.split()
        return " ".join(reversed(words))


if __name__ == "__main__":
    rw = ReverseWords()
    print(rw.reverse_words("Hello World from Python"))
