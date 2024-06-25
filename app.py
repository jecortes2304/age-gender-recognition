import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from routes import routes


app = FastAPI()
app.openapi_version = "3.0.2"

for route in routes:
    app.include_router(route)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
