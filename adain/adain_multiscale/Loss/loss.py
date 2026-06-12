import torch
import torch.nn as nn
from Model.adain_multiscale import calc_mean_std

class StyleTransferLoss(nn.Module):
    def __init__(self, lambda_style: float = 10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_style = lambda_style

    def forward(self, output_feats, content_feats, style_feats):
        # 1. Content Loss: Chỉ tính trên tầng sâu nhất (h4)
        # Vì h4 giữ thông tin cấu trúc/nội dung quan trọng nhất
        c_loss = self.mse(output_feats[3], content_feats[3])

        # 2. Style Loss: Tính trên cả 4 tầng (h1, h2, h3, h4)
        s_loss = 0.0
        for out_f, style_f in zip(output_feats, style_feats):
            out_mean, out_std = calc_mean_std(out_f)
            style_mean, style_std = calc_mean_std(style_f)
            
            s_loss += self.mse(out_mean, style_mean)
            s_loss += self.mse(out_std, style_std)

        # Tổng hợp loss
        total_loss = c_loss + self.lambda_style * s_loss
        return total_loss, c_loss, s_loss

# --- Cách tích hợp vào vòng lặp training ---
# criterion = StyleTransferLoss(lambda_style=10.0)
# optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

# Trong vòng lặp:
# output = model(content, style)
# out_feats = model.encoder(output)
# c_feats = model.encoder(content)
# s_feats = model.encoder(style)
# loss, c_l, s_l = criterion(out_feats, c_feats, s_feats)
# loss.backward()
# optimizer.step()