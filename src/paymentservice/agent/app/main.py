from fastapi import FastAPI
from app.router import router as payment_router
 
app = FastAPI(title="Microservices Agent")

app.include_router(payment_router)