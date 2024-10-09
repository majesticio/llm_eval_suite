# models/model_loader.py

import ollama

class ModelWrapper:
    def __init__(self, model_name, custom_client_host=None):
        self.model_name = model_name
        if custom_client_host:
            # Use custom client
            self.client = ollama.Client(host=custom_client_host)
            self.use_custom_client = True
        else:
            # Use standard module
            self.client = None  # Will use the standard module
            self.use_custom_client = False

    def chat(self, prompt):
        messages = [
            {
                'role': 'user',
                'content': prompt,
            },
        ]
        if self.use_custom_client:
            response = self.client.chat(model=self.model_name, messages=messages)
            content = response['message']['content']
        else:
            response = ollama.chat(model=self.model_name, messages=messages)
            content = response['message']['content']
        return content
