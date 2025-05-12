import os
from glob import glob
import json

splits = ["train", "test"]

for split in splits:
    json_paths = glob(os.path.join("data", split, "json", "*.json"))
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 첫 번째 bounding box 좌표 추출
        pt = data['annotations']['points'][0]
        xtl, ytl = pt['xtl'], pt['ytl']
        xbr, ybr = pt['xbr'], pt['ybr']

        # roi 크기 계산
        width  = xbr - xtl  # x 방향 크기
        height = ybr - ytl  # y 방향 크기

        # annotations 아래에 roi_size 추가
        data['annotations']['roi_size'] = {
            'x': width,
            'y': height
        }

        # 변경된 JSON 파일에 덮어쓰기
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f'x:{width}, y:{height}')