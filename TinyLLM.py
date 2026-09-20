from random import choices

class TinyLLM:
    def fit(self, text):
        self.text = text

    def generate(self, n):
        return "".join(choices(self.text, k=n)) if self.text else ""

model = TinyLLM()
model.fit("hello world hello tiny language model")
print(model.generate(100))
