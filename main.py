"""
FastAPI application for bird counting and weight estimation.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import config

# Create FastAPI app
app = FastAPI(
    title="Bird Counting & Weight Estimation API",
    description="API for detecting, tracking, and counting birds in poultry farm CCTV videos with weight estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("=" * 60)
    print("Bird Counting & Weight Estimation API")
    print("=" * 60)
    print(f"Server starting on {config.API_HOST}:{config.API_PORT}")
    print(f"Outputs directory: {config.OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
