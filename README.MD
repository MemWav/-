# 학습 데이터셋 설정

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=153

에서 이미지와 json파일을 다운받아
```
📦data
 ┣ 📂image
 ┃ ┣ 📜c5_d0_000001.jpg
 ┃ ┣ 📜c5_d0_000002.jpg
 ┃ ┣    ⁞
 ┣ 📂json
 ┃ ┗ 📜c5_d0_000001.json
 ┃ ┗ 📜c5_d0_000002.json
 ┃ ┣    ⁞
```
의 구조로 root directory에 놓기. (* utils/image_resize.py 참고)

# 주의사항

모델 학습 과정에서 nn.CrossEntropyLoss()를 사용하기 때문에 모델 return 부분에 softmax가 적용되어 있지 않음. 추후 정확성 판별 시 softmax 별도 적용 필요.
