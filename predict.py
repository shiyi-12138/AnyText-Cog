"""
AnyText Cog Predictor for Replicate deployment.
Supports text-generation and text-editing modes.
"""

import os
import re
import tempfile
import cv2
import numpy as np
from cog import BasePredictor, Input, Path
from modelscope.pipelines import pipeline

class Predictor(BasePredictor):
    def setup(self):
        font_path = self._get_font_path()
        self.pipe = pipeline(
            "my-anytext-task",
            model="damo/cv_anytext_text_generation_editing",
            model_revision="v1.1.3",
            use_fp16=True,
            use_translator=False,
            font_path=font_path,
        )

    def _get_font_path(self):
        font_paths = [
            "font/Arial_Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
        import subprocess
        try:
            result = subprocess.run(
                ["fc-list", ":lang=en", "file"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                return result.stdout.strip().split("\n")[0].split(":")[0].strip()
        except Exception:
            pass
        return "font/Arial_Unicode.ttf"

    def _count_lines(self, prompt):
        left_q = chr(0x201c)
        right_q = chr(0x201d)
        prompt = prompt.replace(left_q, '"').replace(right_q, '"')
        matches = re.findall(r'"(.*?)"', prompt)
        if len(matches) == 0:
            return 1
        return len(matches)

    def _generate_positions(self, w, h, n_lines, max_trys=200):
        img = np.zeros((h, w, 1), dtype=np.uint8)
        rectangles = []
        attempts = 0
        n_pass = 0
        low_edge = int(max(w, h) * 0.3 if n_lines <= 3 else max(w, h) * 0.2)
        while attempts < max_trys:
            rect_w = min(
                np.random.randint(max((w * 0.5) // n_lines, low_edge), w),
                int(w * 0.8)
            )
            ratio = np.random.uniform(4, 10)
            rect_h = max(low_edge, int(rect_w / ratio))
            rect_h = min(rect_h, int(h * 0.8))
            rotation_angle = 0
            rand_value = np.random.rand()
            if rand_value < 0.7:
                pass
            elif rand_value < 0.8:
                rotation_angle = np.random.randint(0, 40)
            elif rand_value < 0.9:
                rotation_angle = np.random.randint(140, 180)
            else:
                rotation_angle = np.random.randint(85, 95)
            x = np.random.randint(0, w - rect_w)
            y = np.random.randint(0, h - rect_h)
            rect_pts = cv2.boxPoints(
                ((rect_w / 2, rect_h / 2), (rect_w, rect_h), rotation_angle)
            )
            rect_pts = np.int32(rect_pts)
            rect_pts += (x, y)
            if (
                np.any(rect_pts < 0)
                or np.any(rect_pts[:, 0] >= w)
                or np.any(rect_pts[:, 1] >= h)
            ):
                attempts += 1
                continue
            if any(_check_polygon_overlap(rect_pts, rp) for rp in rectangles):
                attempts += 1
                continue
            n_pass += 1
            cv2.fillPoly(img, [rect_pts], 255)
            rectangles.append(rect_pts)
            if n_pass == n_lines:
                break
        return img

    def _check_channels(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] > 3:
            image = image[:, :, :3]
        return image

    def _resize_image(self, img, max_length=768):
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > max_length:
            scale = max_length / max_dim
            w = int(round(w * scale))
            h = int(round(h * scale))
            img = cv2.resize(img, (w, h))
        h, w = img.shape[:2]
        img = cv2.resize(img, (w - (w % 64), h - (h % 64)))
        return img

    def predict(
        self,
        image: Path = Input(
            description="Input image. For text-generation: reference for positioning. For text-editing: the image to edit."
        ),
        prompt: str = Input(
            description="Text prompt. Enclose text to add/edit in double quotes."
        ),
        mode: str = Input(
            default="text-generation",
            choices=["text-generation", "text-editing"],
            description="Mode: text-generation adds text, text-editing modifies existing text.",
        ),
        seed: int = Input(default=-1, description="Random seed (-1 for random)"),
        ddim_steps: int = Input(default=20, description="Number of DDIM sampling steps"),
        image_count: int = Input(default=2, description="Number of output images to generate"),
        cfg_scale: float = Input(default=9.0, description="Classifier-free guidance scale"),
        strength: float = Input(default=1.0, description="Control strength"),
    ) -> list[Path]:
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image from {image}")
        img = self._check_channels(img)
        img = self._resize_image(img)
        h, w = img.shape[:2]
        n_lines = self._count_lines(prompt)
        params = {
            "mode": mode,
            "sort_priority": chr(0x2195),
            "show_debug": False,
            "revise_pos": False,
            "image_count": image_count,
            "ddim_steps": ddim_steps,
            "image_width": w,
            "image_height": h,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "eta": 0.0,
            "a_prompt": "best quality, extremely detailed, 4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
            "n_prompt": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
            "base_model_path": "",
            "lora_path_ratio": "",
        }
        if mode == "text-generation":
            pos_imgs = self._generate_positions(w, h, n_lines)
            input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_imgs,
            }
        elif mode == "text-editing":
            edit_image = img.clip(1, 255)
            pos_imgs = np.zeros((h, w, 1), dtype=np.uint8)
            input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_imgs,
                "ori_image": edit_image,
            }
        results, rtn_code, rtn_warning, debug_info = self.pipe(input_data, **params)
        if rtn_code < 0:
            raise RuntimeError(f"AnyText inference failed: {rtn_warning}")
        output_paths = []
        output_dir = tempfile.mkdtemp()
        for idx, img_result in enumerate(results):
            img_rgb = img_result[..., ::-1]
            out_path = os.path.join(output_dir, f"output_{idx}.png")
            cv2.imwrite(out_path, img_rgb)
            output_paths.append(Path(out_path))
        return output_paths


def _check_polygon_overlap(pts1, pts2):
    poly1 = cv2.convexHull(pts1)
    poly2 = cv2.convexHull(pts2)
    r1 = cv2.boundingRect(poly1)
    r2 = cv2.boundingRect(poly2)
    return (
        r1[0] + r1[2] >= r2[0]
        and r2[0] + r2[2] >= r1[0]
        and r1[1] + r1[3] >= r2[1]
        and r2[1] + r2[3] >= r1[1]
    )
