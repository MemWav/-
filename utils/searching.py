import json
import os


image_folder = "data/train/image"
json_folder = "data/train/json"

# json 폴더에서 disease가 1 이상인(질병이 있는) json 파일을 찾는다.
json_files = []
for root, dirs, files in os.walk(json_folder):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))
            # print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('description', {}).get('disease', 0) >= 1:
                    print(f"질병이 있는 JSON 파일: {os.path.join(root, file)}")
                else:
                    # print(f"질병이 없는 JSON 파일: {os.path.join(root, file)}")
                    pass