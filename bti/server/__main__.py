"""Run with ``python -m bti.server``."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("bti.server.app:app", host=os.getenv("BTI_HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), reload=False)
