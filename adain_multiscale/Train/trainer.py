import torch
import torch.nn as nn

class AdaINTrainer:
    def __init__(self, model, optimizer, lambda_style=10.0,lambda_content = 1.0, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.device = device
        self.mse = nn.MSELoss()

    def _calc_stats(self, feats):
        """Tính mean và std cho một danh sách (tuple) các feature maps."""
        return [calc_mean_std(f) for f in feats]

    def _style_loss(self, out_feats, style_feats):
        """Tính loss trên cả 4 tầng (h1, h2, h3, h4)."""
        loss = 0.0
        out_stats = self._calc_stats(out_feats)
        style_stats = self._calc_stats(style_feats)
        
        for (out_m, out_s), (style_m, style_s) in zip(out_stats, style_stats):
            loss += self.mse(out_m, style_m)
            loss += self.mse(out_s, style_s)
        return loss

    def train_step(self, content, style):
        self.model.train()
        self.optimizer.zero_grad()
        
        content = content.to(self.device)
        style = style.to(self.device)
        
        # 1. Trích xuất đặc trưng
        c_feats = self.model.encoder(content) # Trả về h1, h2, h3, h4
        s_feats = self.model.encoder(style)
        s_stats = self._calc_stats(s_feats)
        
        # 2. Decoder forward với Multi-layer Injection
        output = self.model.decoder(c_feats, s_stats)
        
        # 3. Tính toán Loss
        out_feats = self.model.encoder(output)
        
        c_loss = self.mse(out_feats[3], c_feats[3]) # Content loss tại tầng h4
        s_loss = self._style_loss(out_feats, s_feats) # Style loss đa tầng
        
        total_loss = self.lambda_content * c_loss + self.lambda_style * s_loss
        
        # 4. Update trọng số
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "content_loss": c_loss.item(),
            "style_loss": s_loss.item()
        }

    def validate(self, content, style):
        self.model.eval()
        with torch.no_grad():
            content, style = content.to(self.device), style.to(self.device)
            c_feats = self.model.encoder(content)
            s_feats = self.model.encoder(style)
            s_stats = self._calc_stats(s_feats)
            
            output = self.model.decoder(c_feats, s_stats)
            out_feats = self.model.encoder(output)
            
            c_loss = self.mse(out_feats[3], c_feats[3])
            s_loss = self._style_loss(out_feats, s_feats)
            total_loss = self.lambda_content* c_loss + self.lambda_style * s_loss
            
        return {
            "total_loss": total_loss.item(),
            "content_loss": c_loss.item(),
            "style_loss": s_loss.item()
        }