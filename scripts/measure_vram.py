import sys, torch, json
sys.path.insert(0, r"D:\Хакатон мера\falcon")
from falcon.model import build_model
from falcon.losses import ReIDCriterion


def measure(bs, ckpt, size=256):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build_model(num_classes=1156, embedding_dim=2048, pretrained=False, verbose=False).cuda().train()
    m.set_gradient_checkpointing(ckpt)
    crit = ReIDCriterion(1156, 2048, center_weight=0.0005).cuda()
    opt = torch.optim.Adam(list(m.parameters()) + list(crit.parameters()), lr=3.5e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(bs, 3, size, size, device="cuda")
    y = torch.arange(bs, device="cuda") // 4
    cams = torch.arange(bs, device="cuda") % 4
    import time
    for i in range(4):
        if i == 1: torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            f, lg = m(x)
            loss, _ = crit(f.float(), lg.float(), y, cams)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / 3
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    del m, crit, opt, x, y, cams
    torch.cuda.empty_cache()
    return peak, reserved, per_step


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    print("свободно %.2f ГБ из %.2f ГБ (занято другим: %.2f ГБ)" % (
        free/1e9, total/1e9, (total-free)/1e9))
    print()
    print("%-6s %-14s %-11s %-11s %-10s" % ("батч", "чекпоинтинг", "пик", "зарезерв.", "с/шаг"))
    for ckpt in (False, True):
        for bs in (32, 48, 64, 80):
            try:
                peak, res, step = measure(bs, ckpt)
                print("%-6d %-14s %-11s %-11s %-10.2f" % (
                    bs, "да" if ckpt else "нет", "%.2f ГБ" % peak, "%.2f ГБ" % res, step), flush=True)
            except torch.OutOfMemoryError:
                print("%-6d %-14s не влезает" % (bs, "да" if ckpt else "нет"), flush=True)
                torch.cuda.empty_cache()
