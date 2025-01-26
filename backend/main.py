from fastapi import FastAPI, Query
import dpath.util
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

app = FastAPI(root_path="/api/v1")


@app.get("/places")
def places(query: str = Query(None, title="Query", description="検索クエリ", example="仙台駅周辺で美味しいラーメン屋")):

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
        "places.rating",
        "places.userRatingCount",
        "places.regularOpeningHours.weekdayDescriptions",
        "places.priceRange"
    ])
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }
    
    payload = {
        "textQuery": query,
        "maxResultCount": 10,
        "languageCode": 'ja',
    }

    response = requests.post(url, headers=headers, json=payload)

    return list(map(lambda x: {
        "name": dpath.util.get(x, "name/text", default=None),
        "address": x["formattedAddress"],
        "types": x["types"],
        "website": dpath.util.get(x, "websiteUri", default=None),
        "reviews": dpath.util.get(x, "reviews", default=[]),
        "userReviewCount": dpath.util.get(x, "userRatingCount", default=None),
        "rating": dpath.util.get(x, "rating", default=None),
        "photoUri": get_photo(x["photos"][0]["name"]) if "photos" in x and x["photos"][0] else None,
        "businessHour": dpath.util.get(x, "regularOpeningHours/weekdayDescriptions", default=None),
        "minPrice": get_price(dpath.util.get(x, "priceRange/startPrice", default=None)),
        "maxPrice": get_price(dpath.util.get(x, "priceRange/endPrice", default=None)),
    }, response.json()["places"]))

def get_price(price):
    if price is None:
        return None
    else:
        return f"{price['units']}{price['currencyCode']}"

def get_photo(name: str):
    url = f"https://places.googleapis.com/v1/{name}/media?maxHeightPx=400&maxWidthPx=400&skipHttpRedirect=true"


    headers = {
        "X-Goog-Api-Key": api_key,
    }

    response = requests.get(url, headers=headers)


    return response.json()["photoUri"]
