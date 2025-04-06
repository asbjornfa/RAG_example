Hello there,

This is a simple example of how to setup RAG (Retrieval-Augmented Generation) with FastAPI.

**Setup vitual environment**

If you currently dont have a .venv or base file please use this command to create one.
***This will create a file directory called base with everything***

On Mac:
```
python3 -m venv base
```

On Windows:
```
python -m venv base
```
When the file is created, you can enter the virtual environment like this.

On Mac:
```
source base/bin/activate
```
On Windows:
```
base/Scripts/activate
```

**Setup Ollama LLM**

For this code to work you'll have to have a LLM downloaded from Ollama.
If you want to use the one that is used in side code example, then type the command:
```
ollama run gemma2:9b
```


**Relevant imports**

For relevant imports use this command:
```
pip install -r requirements.txt
```

For adding additional imports to the requirements file use this command:
```
pip freeze > requirements.txt
```

**How to run the application**

This project is using FastApi, with uvicorn.

To run access the api endpoint use this command:

```
uvicorn main:app --reload
```
