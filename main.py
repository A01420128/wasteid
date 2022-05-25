from fastapi import FastAPI

from waste import CLASSES
from identifier import idwaste

app = FastAPI()

@app.get("/identify")
async def root(image: str):
    return idwaste(image, "./model/WasteClassificationModel.h5")
