from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
from app.config import settings
from app.routes import users, auth, stt, ws

app = FastAPI() 
# Allow domain frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # or ["*"] for all with no striction 
    allow_credentials=True, # True: add credentials in request 
    allow_methods=["*"],     # GET, POST, PUT, DELETE…
    allow_headers=["*"],     # header options
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(stt.router, prefix="/api/stt", tags=["stt"])
app.include_router(ws.router, prefix="/api/ws", tags=["ws"])

logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return "Hello, I am alive"

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # host="0.0.0.0": other can access backend by my ip, eg: http://192.168.1.10:8000.