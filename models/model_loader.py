# models/model_loader.py

from ollama import Client

class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = Client(host='http://192.168.5.85:11434')

    def chat(self, prompt):
        response = self.client.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
