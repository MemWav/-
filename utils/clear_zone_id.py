import os

# 현재 디렉토리부터 하위 디렉토리까지 탐색
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(":Zone.Identifier"):
            path = os.path.join(root, file)
            try:
                os.remove(path)
                print(f"삭제됨: {path}")
            except Exception as e:
                print(f"삭제 실패: {path} ({e})")
