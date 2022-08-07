from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn

from transformers import MarianMTModel, MarianTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

BERT_PATH = "E:/AI/NLP/Translator/Saved_model"
MODEL_PATH = "E:/AI/NLP/Translator/Saved_model/model.bin"
CONFIG_FILE = 'E:/AI/NLP/Translator/Saved_model/config.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html',
     context={"request": request,"title": "MACHINE TRANSLATION" })
    
@app.post('/translated_text')
async def mtDanish(request: Request, inputtext: str = Form(...)):    
    src_text = [inputtext]

    tokenizer = MarianTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case = True)

    model = MarianMTModel.from_pretrained(MODEL_PATH, config = CONFIG_FILE, num_labels=3)

    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

    output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    # return templates.TemplateResponse('translated.html', context={"request": request,"corpus": corpus, "result": output})

    return output[0]

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    