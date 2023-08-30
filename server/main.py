from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys
# import gunicorn
# import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from AppStore import AppStore, USER_QUERY, USER_ANSWERS

root_path = str(Path().resolve().parent)

source_paths = [
    root_path,
    root_path + '/recommender',
    root_path + '/server'
]

for source_path in source_paths:
    if source_path not in sys.path:
        sys.path.append(source_path)

from RedditRecommenderSystem import RedditRecommenderSystem


app = FastAPI()

@app.on_event("startup")
def load_recommender():
    global recommender
    global appStore

    recommender = RedditRecommenderSystem()
    recommender.config()
    appStore = AppStore()


# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_version = 1


class RequestIn(BaseModel):
    req: str
    thres: float|None


class ResponseOut(BaseModel):
    movieIds: list


@app.post("/search_recs/", response_model=ResponseOut)
async def search_phrase(payload: RequestIn):
    try:
        kwargs = {}
        if payload.thres:
            kwargs['threshold'] = payload.thres

        appStore.addToStore({ 'type': USER_QUERY, 'value': payload.req })
        ids, candidates = recommender.predict_for_request(payload.req, **kwargs)
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400)

    appStore.addToStore({ 'type': USER_ANSWERS, 'value': {'query': payload.req, 'candidates': [c['title'] for c in candidates] }})
    return {"movieIds": ids}


@app.get("/")
async def home():
    return {"health_check": "OK", "model_version": model_version}
