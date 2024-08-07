# 《精通transformer》 CH10

# 1. FastAPI Transformer模型服务
# --------------main.py-----------
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

model_name = 'distilbert-base-cased-distilled-squad'
model = pipeline(model=model_name, tokenizer=model_name, task='question-answering')

class QADataModel(BaseModel):
    question: str
    context: str

app = FastAPI()

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(question = input_data.question, context=input_data.context)
    return {"answer": result["answer"]}

if __name__ == '__main__':
    uvicorn.run('main:app', workers=1)

# run: python main.py,  then使用Postman测试

# 2. 容器化API
# 2.1 将main.py文件放在app目录, main.py删除 if __name__=='__main__':uvicorn.run('main:app', workers=1)
# 2.2 为FastAPI构建Dockerfile文件
# dockerfile文件内容如下：
'''
FROM python:3.7

RUN pip install torch

RUN pip install fastapi uvicorn transformers

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
# 2.3 构建Docker容器
'''
$ docker build -t qaapi .
$ docker run -p 8005:8000 qaapi

'''
# 3. 使用TFX提供更快的Transformer模型服务
# 3.1 TFModel
from transformers import TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-imdb", from_pt=True)
model.save_pretrained("tfx_model", saved_model=True)
# 3.2 启动TFX服务
'''
$ docker pull tensorflow/serving
$ docker run -d --name serving_base tensorflow/serving
$ docker cp tfx_model/saved_model tfx:/models/bert
$ docker commit --change "ENV MODEL_NAME bert" tfx my_bert_model
$ docker kill tfx
$ docker run -p 8501:8501 -p 8500:8500 --name  my_bert_model

'''
# -----------main.py---------
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertConfig
import requests
import json
import numpy as np

tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")
config = BertConfig.from_pretrained("nateraw/bert-base-uncased-imdb")


class DataModel(BaseModel):
    text: str

app = FastAPI()

@app.post("/sentiment")
async def sentiment_analysis(input_data: DataModel):
    print(input_data.text)
    tokenized_sentence = [dict(tokenizer(input_data.text))]
    data_send = {"instances": tokenized_sentence}
    response = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(data_send))
    result = np.abs(json.loads(response.text)["predictions"][0])
    return {"sentiment": config.id2label[np.argmax(result)]}


if __name__ == '__main__': 
     uvicorn.run('main:app', workers=1) 
