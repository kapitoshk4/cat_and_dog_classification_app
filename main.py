import os.path
import uuid

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.templating import _TemplateResponse

from app.classifier import classify
from app.config import UPLOAD_FOLDER, cnn_model

app = FastAPI()

template = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def upload_file_page(request: Request) -> _TemplateResponse:
    return template.TemplateResponse("upload_file.html", {"request": request})


@app.post("/predict/")
async def upload_file(request: Request, file: UploadFile = File(...)) -> _TemplateResponse:
    file.filename = f"{uuid.uuid4()}.jpg"
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    contents = await file.read()

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with open(upload_image_path, "wb") as f:
        f.write(contents)

    label, prob = classify(cnn_model, upload_image_path)
    prob = round(prob * 100, 2)

    return template.TemplateResponse(
        "prediction.html",
        {
            "request": request,
            "label": label,
            "probability": prob,
            "image": file.filename
        }
    )
