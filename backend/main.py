from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from upload import router as upload_router

app = FastAPI(
    title="BusinessGuide AI",
    description="AI-powered Business Decision Support System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)


@app.get("/")
def home():
    return {
        "message": "BusinessGuide AI backend is running"
    }