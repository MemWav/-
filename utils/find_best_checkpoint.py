import os
import re

def find_best_checkpoint(ckpt_dir):
    # 1) acc 패턴으로 가장 높은 정확도 파일 탐색
    best_acc = -1.0
    best_path = None
    pattern = re.compile(r'acc([0-9]+\.[0-9]+)')
    pth_files = []

    for fn in os.listdir(ckpt_dir):
        if not fn.endswith('.pth'):
            continue
        full_path = os.path.join(ckpt_dir, fn)
        pth_files.append(full_path)

        m = pattern.search(fn)
        if m:
            acc = float(m.group(1))
            if acc > best_acc:
                best_acc = acc
                best_path = full_path

    # 2) acc 패턴 파일이 없으면, 가장 최근 수정된 .pth 파일 선택
    if best_path is None and pth_files:
        # 수정시간 기준 내림차순 정렬 후 첫 번째
        pth_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        best_path = pth_files[0]
        print(f">> No acc-pattern checkpoint found. Loading latest checkpoint: "
              f"{os.path.basename(best_path)}")

    # 3) .pth 파일 자체가 없으면 에러 발생
    if best_path is None:
        raise FileNotFoundError(f"No .pth files found in {ckpt_dir}")

    print(f">> Loading best checkpoint: {os.path.basename(best_path)}"
          + (f" (acc={best_acc:.4f})" if best_acc >= 0 else ""))
    return best_path
