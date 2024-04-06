from pymongo import MongoClient

client = MongoClient("mongodb+srv://supportek:loRyzvrMM0ScEt4Z@cluster0.08cunpc.mongodb.net/admin?replicaSet=atlas"
                     "-bbql02-shard-0&readPreference=primary&srvServiceName=mongodb&connectTimeoutMS=10000&authSource"
                     "=admin&authMechanism=SCRAM-SHA-1")
db = client["supportekdb"]  # Replace "your_database_name" with your database name
knowledge_sources_collection = db["knowledge_sources"]
