import copy
import torch
from torch.nn.utils import clip_grad_norm_
from src.losses.diffusion_loss import style_diffusion_loss

class DiffusionTrainer:
    """
    Lớp điều phối quá trình huấn luyện cho Style-guided Diffusion Model.
    Quản lý Model chính, Model EMA, Optimizer và tính toán Loss.
    """
    def __init__(self, model: torch.nn.Module, style_encoder: torch.nn.Module, 
                 scheduler: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                 device: str, grad_clip: float = 1.0, ema_decay: float = 0.9999, 
                 loss_weights: dict = None):
        
        self.model = model
        self.style_encoder = style_encoder
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        
        # Thiết lập trọng số Loss mặc định nếu không được truyền vào
        if loss_weights is None:
            self.loss_weights = {"noise": 1.0, "style": 0.1, "content": 0.01}
        else:
            self.loss_weights = loss_weights
            
        # =========================================================
        # BẢO VỆ STYLE ENCODER: KHÔNG BAO GIỜ NHẬN GRADIENT
        # VGG chỉ dùng để trích xuất feature, việc cập nhật weights 
        # sẽ phá hỏng các đặc trưng ImageNet đã được pre-train.
        # =========================================================
        self.style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False
            
        # =========================================================
        # KHỞI TẠO EMA MODEL (Exponential Moving Average)
        # Trong Diffusion, mô hình chính dao động rất mạnh khi train.
        # Ta giữ một bản sao EMA cập nhật chậm rãi để sinh ảnh mượt hơn.
        # =========================================================
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval() # EMA model chỉ dùng để inference, luôn ở chế độ eval
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def _update_ema(self):
        """
        Cập nhật trọng số của EMA model dựa trên trọng số của Model chính.
        Công thức: EMA_weight = decay * EMA_weight + (1 - decay) * Model_weight
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                # Chỉ cập nhật nhữn tham số thực sự tham gia vào quá trình học
                if model_param.requires_grad:
                    ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1.0 - self.ema_decay)

    def train_step(self, batch: dict) -> dict:
        """
        Thực hiện một bước huấn luyện (Forward, Backward, Optimizer step).
        """
        # Đảm bảo model ở chế độ train (bật lại Dropout nếu có)
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Tính toán Loss (Bao gồm Noise, Style, và Content)
        total_loss, info = style_diffusion_loss(
            model=self.model,
            style_encoder=self.style_encoder,
            scheduler=self.scheduler,
            batch=batch,
            weights=self.loss_weights,
            device=self.device
        )
        
        # 2. Lan truyền ngược
        total_loss.backward()
        
        # 3. Gradient Clipping (Chống hiện tượng bùng nổ gradient - Exploding Gradients)
        # Hàm clip_grad_norm_ trả về tổng chuẩn norm của gradient để theo dõi
        grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
        
        # 4. Cập nhật trọng số
        self.optimizer.step()
        
        # 5. Cập nhật bản sao EMA
        self._update_ema()
        
        # Trả về các thông số để log lên TensorBoard / W&B
        return {
            "train_loss": info["total_loss"],
            "noise_loss": info["noise_loss"],
            "style_loss": info["style_loss"],
            "content_loss": info["content_loss"],
            "grad_norm": grad_norm
        }

    def val_step(self, batch: dict) -> dict:
        """
        Thực hiện một bước validation để đánh giá mô hình.
        """
        self.model.eval()
        with torch.no_grad():
            # Tại bước val_step, ta chỉ cần Forward pass, không cập nhật trọng số
            _, info = style_diffusion_loss(
                model=self.model,
                style_encoder=self.style_encoder,
                scheduler=self.scheduler,
                batch=batch,
                weights=self.loss_weights,
                device=self.device
            )
            
        return {
            "val_loss": info["total_loss"],
            "noise_loss": info["noise_loss"],
            "style_loss": info["style_loss"],
            "content_loss": info["content_loss"]
        }

    def save_checkpoint(self, path: str, epoch: int):
        """Lưu lại trạng thái đầy đủ của quá trình huấn luyện."""
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_weights": self.loss_weights
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Khôi phục quá trình huấn luyện từ file checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        
        # Ghi đè lại trọng số loss nếu có trong checkpoint
        if "loss_weights" in ckpt:
            self.loss_weights = ckpt["loss_weights"]
            
        return ckpt["epoch"]