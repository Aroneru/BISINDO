import argparse
import importlib
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None

try:
    import cv2
except ImportError:
    cv2 = None


def parse_size(value: str) -> Tuple[int, int]:
    """Parse image size from 'N' or 'HxW'."""
    if "x" in value.lower():
        h, w = value.lower().split("x", 1)
        return int(h), int(w)
    n = int(value)
    return n, n


def load_labels(label_path: Optional[str]) -> Optional[List[str]]:
    if not label_path:
        return None

    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {path}")

    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels or None


def index_to_letters(index: int) -> str:
    """Convert zero-based class index to alphabetic code: 0->A, 25->Z, 26->AA."""
    if index < 0:
        return "UNKNOWN"

    n = index
    chars = []
    while True:
        n, r = divmod(n, 26)
        chars.append(chr(ord("A") + r))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def get_label_for_index(cls_idx: int, labels: Optional[List[str]]) -> str:
    if labels and 0 <= cls_idx < len(labels):
        return labels[cls_idx]

    # Fallback label to avoid showing raw numeric class id.
    return f"BISINDO {index_to_letters(cls_idx)}"


def build_model(module_name: str, class_name: str, num_classes: Optional[int]) -> nn.Module:
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    if num_classes is None:
        model = model_class()
    else:
        model = model_class(num_classes=num_classes)

    if not isinstance(model, nn.Module):
        raise TypeError("Objek model yang dibuat bukan turunan torch.nn.Module")

    return model


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "model.")
    return state_dict


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    candidate_keys = [
        "fc.weight",
        "classifier.1.weight",
        "classifier.6.weight",
        "classifier.weight",
        "head.weight",
    ]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    return None


def build_torchvision_candidate_models(num_classes: int) -> Dict[str, nn.Module]:
    if tv_models is None:
        return {}

    candidates = {}
    constructors = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
        "mobilenet_v2": tv_models.mobilenet_v2,
        "efficientnet_b0": tv_models.efficientnet_b0,
        "densenet121": tv_models.densenet121,
        "vgg16": tv_models.vgg16,
        "alexnet": tv_models.alexnet,
    }

    for name, ctor in constructors.items():
        try:
            candidates[name] = ctor(weights=None, num_classes=num_classes)
        except TypeError:
            try:
                candidates[name] = ctor(num_classes=num_classes)
            except Exception:
                pass
        except Exception:
            pass

    return candidates


class _VGGContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self._make_vgg19_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    @staticmethod
    def _make_vgg19_features() -> nn.Sequential:
        # VGG19 config "E" to match standard torchvision layer naming.
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = int(v)
        return nn.Sequential(*layers)


class AutoBisindoVGGGRU(nn.Module):
    def __init__(self, gru_input_size: int, gru_hidden_size: int, num_classes: int):
        super().__init__()
        self.vgg19 = _VGGContainer()
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg19.features(x)
        x = self.vgg19.avgpool(x)
        x = torch.flatten(x, 1)

        if x.shape[1] != self.gru.input_size:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.gru.input_size).squeeze(1)

        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def try_build_bisindo_vgg_gru_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[nn.Module]:
    has_vgg = any(k.startswith("vgg19.features.") for k in state_dict.keys())
    has_gru = any(k.startswith("gru.") for k in state_dict.keys())
    has_fc = "fc.weight" in state_dict
    if not (has_vgg and has_gru and has_fc):
        return None

    gru_w = state_dict.get("gru.weight_ih_l0")
    fc_w = state_dict.get("fc.weight")
    if not isinstance(gru_w, torch.Tensor) or not isinstance(fc_w, torch.Tensor):
        return None

    if gru_w.ndim != 2 or fc_w.ndim != 2:
        return None

    gru_hidden_size = int(gru_w.shape[0] // 3)
    gru_input_size = int(gru_w.shape[1])
    num_classes = int(fc_w.shape[0])

    if gru_hidden_size <= 0 or gru_input_size <= 0 or num_classes <= 0:
        return None

    print(
        f"[INFO] Auto arsitektur BISINDO terdeteksi: VGG19+GRU+FC "
        f"(input_size={gru_input_size}, hidden_size={gru_hidden_size}, classes={num_classes})"
    )
    return AutoBisindoVGGGRU(gru_input_size=gru_input_size, gru_hidden_size=gru_hidden_size, num_classes=num_classes)


def try_auto_build_model_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[nn.Module]:
    bisindo_model = try_build_bisindo_vgg_gru_from_state_dict(state_dict)
    if bisindo_model is not None:
        return bisindo_model

    num_classes = infer_num_classes_from_state_dict(state_dict)
    if num_classes is None:
        return None

    candidates = build_torchvision_candidate_models(num_classes)
    if not candidates:
        return None

    best_name = None
    best_model = None
    best_score = -1

    for name, model in candidates.items():
        model_sd = model.state_dict()
        score = 0
        for key, tensor in state_dict.items():
            target = model_sd.get(key)
            if target is not None and tuple(target.shape) == tuple(tensor.shape):
                score += 1

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_model is None or best_score <= 0:
        return None

    model_sd = best_model.state_dict()
    filtered_sd = {
        k: v
        for k, v in state_dict.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }

    if not filtered_sd:
        return None

    print(f"[INFO] Auto model terdeteksi: {best_name} (matched keys: {best_score})")
    return best_model


def load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    model_sd = model.state_dict()
    filtered_sd = {
        k: v
        for k, v in state_dict.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

    dropped = len(state_dict) - len(filtered_sd)
    if dropped > 0:
        print(f"[WARNING] Ada {dropped} parameter checkpoint yang tidak cocok shape dan dilewati.")

    return missing, unexpected


def load_model(
    model_path: str,
    device: torch.device,
    model_module: Optional[str],
    model_class: Optional[str],
    num_classes: Optional[int],
) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        state_dict = extract_state_dict(ckpt)
        if state_dict is None:
            raise ValueError(
                "Format checkpoint tidak dikenali. Gunakan --model-module dan --model-class "
                "jika file berisi state_dict."
            )

        state_dict = normalize_state_dict_keys(state_dict)

        if model_module and model_class:
            model = build_model(model_module, model_class, num_classes)
        else:
            model = try_auto_build_model_from_state_dict(state_dict)
            if model is None:
                raise ValueError(
                    "Checkpoint terdeteksi sebagai state_dict, tapi arsitektur model tidak bisa ditentukan otomatis. "
                    "Isi --model-module dan --model-class, atau install torchvision untuk auto-detect model umum."
                )

        missing, unexpected = load_state_dict_flexible(model, state_dict)
        if missing:
            print(f"[WARNING] Missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()
    return model


def preprocess_array(
    arr: np.ndarray,
    image_size: Tuple[int, int],
    channels: int,
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    if channels not in (1, 3):
        raise ValueError("--channels hanya boleh 1 atau 3")

    if channels == 1:
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        arr = np.expand_dims(arr, axis=-1)
    else:
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    return tensor


def preprocess_image(
    image_path: str,
    image_size: Tuple[int, int],
    channels: int,
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(image_path)
    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    return preprocess_array(arr, image_size, channels, mean, std, device)


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if y.ndim == 1:
            y = y.unsqueeze(0)
    return y


def run_camera_app(
    model: nn.Module,
    device: torch.device,
    image_size: Tuple[int, int],
    channels: int,
    mean: float,
    std: float,
    labels: Optional[List[str]],
    camera_id: int,
    topk: int,
    infer_every_n_frames: int,
):
    if cv2 is None:
        raise ImportError("OpenCV belum terpasang. Install dulu: pip install opencv-python")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera dengan id {camera_id} tidak bisa dibuka")

    frame_count = 0
    last_values = None
    last_indices = None
    fps = 0.0
    last_tick = time.perf_counter()

    print("Kamera aktif. Tekan 'q' untuk keluar.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = max(1e-6, now - last_tick)
        last_tick = now
        inst_fps = 1.0 / dt
        fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

        frame_count += 1

        if frame_count % max(1, infer_every_n_frames) == 0 or last_values is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32) / 255.0

            x = preprocess_array(arr, image_size, channels, mean, std, device)
            logits = predict(model, x)
            probs = torch.softmax(logits, dim=1)

            k = max(1, min(topk, probs.shape[1]))
            last_values, last_indices = torch.topk(probs, k=k, dim=1)

        text_lines = []
        top_name = "Mendeteksi..."
        top_conf = 0.0
        if last_values is not None and last_indices is not None:
            first_idx = int(last_indices[0][0].item())
            first_score = float(last_values[0][0].item())
            top_name = get_label_for_index(first_idx, labels)
            top_conf = first_score

            for rank, (score, cls_idx) in enumerate(
                zip(last_values[0].tolist(), last_indices[0].tolist()), start=1
            ):
                name = get_label_for_index(int(cls_idx), labels)
                text_lines.append(f"{rank}. {name} ({score:.2%})")

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Header panel.
        cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)

        # Right info panel for top-k predictions.
        panel_x0 = max(0, w - 360)
        cv2.rectangle(overlay, (panel_x0, 120), (w, min(h, 120 + 40 * max(1, len(text_lines)) + 20)), (15, 15, 15), -1)

        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, "BISINDO Realtime", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Device: {device}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, "Tekan q untuk keluar", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(frame, top_name, (20, min(h - 40, 160)), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        bar_x, bar_y = 20, min(h - 20, 175)
        bar_w, bar_h = 300, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
        fill_w = int(bar_w * max(0.0, min(1.0, top_conf)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 215, 255), -1)
        cv2.putText(frame, f"Confidence: {top_conf:.2%}", (bar_x + 310, bar_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        y = 150
        for line in text_lines:
            cv2.putText(frame, line, (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
            y += 32

        cv2.imshow("BISINDO Realtime", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Gunakan model BISINDO (.pth) untuk inferensi gambar klasifikasi."
    )
    parser.add_argument("--model-path", default="Bisindo.pth", help="Path file model .pth")
    parser.add_argument("--image", default=None, help="Path gambar input (mode sekali prediksi)")
    parser.add_argument("--camera", action="store_true", help="Aktifkan mode kamera realtime")
    parser.add_argument("--camera-id", type=int, default=0, help="ID kamera, default 0")
    parser.add_argument(
        "--infer-every-n-frames",
        type=int,
        default=3,
        help="Lakukan inferensi setiap N frame untuk performa realtime",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--image-size", default="224", help="Ukuran gambar: N atau HxW, contoh 224 / 224x224")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--mean", type=float, default=0.5, help="Normalisasi mean")
    parser.add_argument("--std", type=float, default=0.5, help="Normalisasi std")

    parser.add_argument("--labels", default=None, help="Path file label (satu label per baris)")
    parser.add_argument("--topk", type=int, default=3, help="Tampilkan top-k prediksi")

    parser.add_argument("--model-module", default=None, help="Module Python yang berisi class model")
    parser.add_argument("--model-class", default=None, help="Nama class model")
    parser.add_argument("--num-classes", type=int, default=None, help="Jumlah kelas saat inisialisasi model")

    args = parser.parse_args()

    if not args.image and not args.camera:
        args.camera = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_size = parse_size(args.image_size)
    labels = load_labels(args.labels)

    model = load_model(
        model_path=args.model_path,
        device=device,
        model_module=args.model_module,
        model_class=args.model_class,
        num_classes=args.num_classes,
    )

    if args.camera:
        print(f"Device: {device}")
        run_camera_app(
            model=model,
            device=device,
            image_size=image_size,
            channels=args.channels,
            mean=args.mean,
            std=args.std,
            labels=labels,
            camera_id=args.camera_id,
            topk=args.topk,
            infer_every_n_frames=args.infer_every_n_frames,
        )
        return

    if not args.image:
        raise ValueError("Isi --image untuk mode gambar, atau gunakan --camera untuk mode realtime")

    x = preprocess_image(
        image_path=args.image,
        image_size=image_size,
        channels=args.channels,
        mean=args.mean,
        std=args.std,
        device=device,
    )

    logits = predict(model, x)
    probs = torch.softmax(logits, dim=1)

    topk = max(1, min(args.topk, probs.shape[1]))
    values, indices = torch.topk(probs, k=topk, dim=1)

    print(f"Device: {device}")
    print(f"Image: {args.image}")
    print("Top prediksi:")

    for rank, (score, cls_idx) in enumerate(zip(values[0].tolist(), indices[0].tolist()), start=1):
        name = get_label_for_index(int(cls_idx), labels)
        print(f"{rank}. word={name} prob={score:.4f}")


if __name__ == "__main__":
    main()
