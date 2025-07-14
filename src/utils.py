import torch


class BackdoorCrossStamp:
    def __init__(
        self,
        image_shape: tuple,
        cross_size: int,
        pos: tuple,
        color: tuple,
        line_width: int,
    ):
        self.image_shape = image_shape
        self.cross_size = cross_size
        self.pos = pos
        self.base_color = torch.tensor(color).view(-1, 1, 1)
        self.line_width = line_width

        self.is_grayscale = len(image_shape) == 2 or image_shape[0] == 1
        if self.is_grayscale:
            self.c = 1
            self.h, self.w = image_shape[-2], image_shape[-1]
        else:
            self.c, self.h, self.w = image_shape

    def stamp(self, img: torch.Tensor) -> torch.Tensor:
        out = img.clone()
        if len(out.size()) == 2:
            out = out.unsqueeze(0)

        x0, y0 = self.pos
        size = self.cross_size
        lw = self.line_width

        xc = x0 + size // 2
        yc = y0 + size // 2

        x_start, x_end = max(x0, 0), min(x0 + size, self.w)
        y_start, y_end = max(y0, 0), min(y0 + size, self.h)

        h_y_start = max(yc - lw // 2, y0)
        h_y_end = min(yc + (lw + 1) // 2, y0 + size)

        v_x_start = max(xc - lw // 2, x0)
        v_x_end = min(xc + (lw + 1) // 2, x0 + size)

        out[:, h_y_start:h_y_end, x_start:x_end] = self.base_color
        out[:, y_start:y_end, v_x_start:v_x_end] = self.base_color

        return out


def evaluate_efficacy(
    model: torch.nn.Module,
    backdoor_dataloader: torch.utils.data.DataLoader ,
) -> float:
    """Evaluates the efficacy of a backdoor attack on the model.
    Data should be a backdoored dataset. Labels should be the attacker target.
    """
    model.eval()
    with torch.no_grad():
        backdoor_right, backdoor_total = 0, 0
        
        for data, target in backdoor_dataloader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            backdoor_right += pred.eq(target.view_as(pred)).sum().item()
            backdoor_total += len(target)
        
    efficacy = backdoor_right / backdoor_total if backdoor_total > 0 else 0
    return efficacy
