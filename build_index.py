import os
import glob
import time
import uuid
import tqdm
import torch
import random
import argparse
from PIL import Image
from torchvision import transforms
from model.AlexNet import AlexNet
from utils.dataset import build_dataloader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest


ap = argparse.ArgumentParser()
ap.add_argument(
    "--dataset_path",
    required=True,
    help="database path containing directory for each class with n images",
)
ap.add_argument(
    "--model_path",
    required=False,
    default="./weights/model_best.pth",
    help="model weights path",
)
ap.add_argument(
    "--image_size",
    required=False,
    type=int,
    default=128,
    help="input image size, defaults to 128",
)
ap.add_argument(
    "--embedding_size",
    required=False,
    type=int,
    default=64,
    help="embedding size from model output",
)
ap.add_argument(
    "--batch_size",
    required=False,
    default=32,
    type=int,
    help="batch size for loading images",
)
ap.add_argument(
    "--overwrite",
    required=False,
    default=False,
    action="store_true",
    help="by default index are updated, use this flag to overwrite existing indexes",
)
ap.add_argument(
    "--collection_name",
    required=False,
    default="bird_species_alexnet_64d",
    action="store_true",
    help="collection name for reading/writing indexes",
)
args = ap.parse_args()


IMAGE_SIZE = args.image_size
EMBEDDING_SIZE = args.embedding_size
BATCH_SIZE = args.batch_size
MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path
COLLECTION_NAME = args.collection_name
IMAGES_PATH = glob.glob(DATASET_PATH+"/*/*")
DEVICE = torch.device("cuda:0")
OVERWRITE = args.overwrite


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


dataset, dataloader = build_dataloader(
    batch_size=BATCH_SIZE,
    root_dir=DATASET_PATH,
    transform=transform,
    shuffle=False,
    num_workers=4,
)


model = AlexNet(input_size=IMAGE_SIZE, embedding_size=EMBEDDING_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Initialize qdrant client
client = QdrantClient(host="localhost", port=6333)
client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True
)

# check if collection exists
existing_collections = [c.name for c in client.get_collections().collections]

# create collection if  doesn't exists
if not (COLLECTION_NAME in existing_collections):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )

total_records_qdrant = client.http.collections_api.get_collection(
    COLLECTION_NAME
).dict()["result"]["vectors_count"]

print(f"Total records inside Qdrant: {total_records_qdrant}")


if OVERWRITE:
    # if you want to re-write the entire index, delete old collection and re-create it.
    print(f"Recreating collection: {COLLECTION_NAME}")
    client.delete_collection(collection_name=COLLECTION_NAME)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )

    total_records_qdrant = client.http.collections_api.get_collection(
        COLLECTION_NAME
    ).dict()["result"]["vectors_count"]

    print(f"Total records inside Qdrant: {total_records_qdrant}")


print(f"batch size: {BATCH_SIZE}, total images to index: {len(glob.glob(DATASET_PATH+'/*/*'))}")

st = time.time()
for batch_idx, (images_batch, labels_batch) in enumerate(tqdm.tqdm(dataloader)):
    with torch.no_grad():
        embeddings = model(images_batch.to(DEVICE)).cpu().numpy()

    points = []
    for embedding, label in zip(embeddings, labels_batch):
        idx = label.item()
        embedding = embedding.tolist()
        label = dataset.label_decode[idx]

        payload_id = uuid.uuid1().int >> 64
        points.append(
            PointStruct(
                id=payload_id,
                payload={"label": label},
                vector=embedding,
            )
        )
    
    # Insert new embeding vector and it's label inside qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

total_records_qdrant = client.http.collections_api.get_collection(
    COLLECTION_NAME
).dict()["result"]["vectors_count"]

print(f"Elapsed: {time.time() - st} seconds")
print(f"Total records inside Qdrant: {total_records_qdrant}")