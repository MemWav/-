import json
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

class AnnotationVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Annotation Visualizer")
        master.geometry("800x600")

        self.image_folder = ""
        self.json_folder = ""
        self.canvas_image = None  # PhotoImage 레퍼런스 유지용
        self.canvas_width = None
        self.canvas_height = None

        # 상단 버튼 프레임
        btn_frame = tk.Frame(master)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(
            btn_frame, text="Select Image Folder", command=self.select_image_folder
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame, text="Select JSON Folder", command=self.select_json_folder
        ).pack(side=tk.LEFT, padx=5)

        # 좌우 분할된 패널
        content = tk.PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 좌측: 이미지 리스트
        self.left_frame = tk.Frame(content)
        content.add(self.left_frame, width=200)
        self.listbox = tk.Listbox(self.left_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # 우측: Canvas에 이미지 표시
        self.right_frame = tk.Frame(content)
        content.add(self.right_frame, stretch="always")
        self.canvas = tk.Canvas(self.right_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Canvas 리사이즈 감지
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def select_image_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self.image_folder = folder
        self._populate_image_list()

    def select_json_folder(self):
        folder = filedialog.askdirectory(title="Select JSON Folder")
        if folder:
            self.json_folder = folder

    def _populate_image_list(self):
        self.listbox.delete(0, tk.END)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(self.image_folder, ext)))
        img_paths.sort()
        for p in img_paths:
            self.listbox.insert(tk.END, os.path.basename(p))

    def on_canvas_resize(self, event):
        # Canvas 크기 갱신
        self.canvas_width = event.width
        self.canvas_height = event.height

    def on_image_select(self, event):
        if not self.json_folder:
            messagebox.showerror("Error", "먼저 JSON 폴더를 선택하세요!")
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        image_name = self.listbox.get(sel[0])
        image_path = os.path.join(self.image_folder, image_name)

        # 매칭되는 JSON 찾기
        matched = None
        for jf in glob.glob(os.path.join(self.json_folder, "*.json")):
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('description', {}).get('image') == image_name:
                matched = data
                break

        if not matched:
            messagebox.showwarning("Warning", f"해당 이미지에 매칭되는 JSON이 없습니다:\n{image_name}")
            return

        # PIL 이미지 로드
        pil_img = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_img.size

        # Canvas 크기 얻기(최초엔 None일 수 있어서 강제 업데이트)
        if self.canvas_width is None or self.canvas_height is None:
            self.canvas.update_idletasks()
            self.canvas_width = self.canvas.winfo_width()
            self.canvas_height = self.canvas.winfo_height()

        # 스케일 계산 (비율 유지, 최대 1.0)
        scale = min(self.canvas_width / img_w, self.canvas_height / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # 이미지 리사이즈 후 바운딩 박스 그리기
        # 수정된 코드
        pil_resized = pil_img.resize(
            (new_w, new_h),
            # Pillow 10 이상
            Image.Resampling.LANCZOS
            # 또는 하위 호환을 위해
            # Image.LANCZOS
        )
        draw = ImageDraw.Draw(pil_resized)
        for pt in matched.get('annotations', {}).get('points', []):
            xtl = pt['xtl'] * scale
            ytl = pt['ytl'] * scale
            xbr = pt['xbr'] * scale
            ybr = pt['ybr'] * scale
            draw.rectangle((xtl, ytl, xbr, ybr), outline='red', width=3)

        # Canvas에 표시
        tk_img = ImageTk.PhotoImage(pil_resized)
        self.canvas_image = tk_img
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationVisualizer(root)
    root.mainloop()
