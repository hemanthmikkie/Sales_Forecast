"""
Module 10 & 12: FastAPI Main Application
Application entrypoint, lifespan startup events, table initialization,
model cache warm-up, and root status routing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from database.database import engine, Base, SessionLocal
from database.crud import seed_master_data
from api.routes import router as api_router, load_ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for database initialization and model loading."""
    print("[Startup] Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    print("[Startup] Checking master data (seeding only if products table is empty)...")
    db = SessionLocal()
    try:
        seed_master_data(db)
    finally:
        db.close()

    print("[Startup] Loading ML demand forecasting artifacts...")
    load_ml_models()
    print("[Startup] Application ready — visit http://localhost:8000/docs")
    yield
    print("[Shutdown] Cleaning up resources...")


app = FastAPI(
    title="Sales Demand Forecasting & Inventory Optimization API",
    description=(
        "Production REST API for retail demand forecasting, inventory risk evaluation, "
        "and automated reorder quantity recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend or dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
# Include without prefix for direct /health, /sales, /forecast compatibility as specified
app.include_router(api_router)
# Also include with /api/v1 prefix for modern versioned routing
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects base root to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
