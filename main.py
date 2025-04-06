from fastapi import FastAPI

import Rag1

app = FastAPI()


app.include_router(Rag1.router, prefix="/Rag1", tags=["Rag1"])
