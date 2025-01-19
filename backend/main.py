from fastapi import FastAPI
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

app = FastAPI(root_path="/api/v1")


@app.get("/places")
def places(query: str):

    response = search_text_query(query)

    return response


def search_text_query(query: str):
    url = "https://places.googleapis.com/v1/places:searchText"
    
    # 環境変数から API キーを取得
    
    field_mask = ",".join([
        "places.displayName",
        "places.formattedAddress",
        "places.types",
        "places.websiteUri",
        "places.reviews.text.text",
        "places.reviews.rating",
        "places.photos.name",
        "places.rating"
    ])
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    
    payload = {
        "textQuery": query,
        "maxResultCount": 20,
        "languageCode": 'ja',
    }

    response = requests.post(url, headers=headers, json=payload)

    return list(map(lambda x: {
        "name": x["displayName"],
        "address": x["formattedAddress"],
        "types": x["types"],
        "website": x["websiteUri"] if "websiteUri" in x else None,
        "reviews": x["reviews"] if "reviews" in x else None,
        "rating": x["rating"] if "rating" in x else None,
        "photoUri": get_photo(x["photos"][0]["name"]) if "photos" in x and x["photos"][0] else None
    }, response.json()["places"]))


def get_photo(name: str):
    url = f"https://places.googleapis.com/v1/{name}/media?maxHeightPx=400&maxWidthPx=400&skipHttpRedirect=true"


    headers = {
        "X-Goog-Api-Key": api_key,
    }

    response = requests.get(url, headers=headers)


    return response.json()["photoUri"]
