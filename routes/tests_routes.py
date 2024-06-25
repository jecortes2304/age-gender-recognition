from fastapi import APIRouter, FastAPI, WebSocket

router_tests = APIRouter(
    tags=["Tests"],
    prefix="/tests",
)


@router_tests.get("/")
async def root():
    return {"message": "Hello World"}
