import pymongo
import pandas as pd
import os

datalink = os.environ.get("DATABASE_LINK")
client = pymongo.MongoClient(datalink)
db = client["Heartpatientdatabase"]
col = db["Heart_Data"]
df = pd.DataFrame(list(col.find({}, {'_id': False})))
print("MongoDB columns:", df.columns.tolist())
