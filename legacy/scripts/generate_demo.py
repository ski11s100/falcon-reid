"""Generate small synthetic images for a smoke-test demonstration."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

OUT = Path(__file__).resolve().parent.parent / "demo_images"
OUT.mkdir(exist_ok=True)
DATASET = Path(__file__).resolve().parent.parent / "work" / "demo_dataset"


def car(color: str, shift: int = 0) -> Image.Image:
    image = Image.new("RGB", (640, 400), "#98a7b5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 270, 640, 400), fill="#34383d")
    draw.ellipse((105 + shift, 245, 205 + shift, 345), fill="#151515", outline="#999", width=9)
    draw.ellipse((435 + shift, 245, 535 + shift, 345), fill="#151515", outline="#999", width=9)
    draw.rounded_rectangle((80 + shift, 160, 560 + shift, 300), radius=45, fill=color, outline="#202020", width=6)
    draw.polygon([(190 + shift,160),(265 + shift,92),(430 + shift,92),(500 + shift,160)], fill=color, outline="#202020")
    draw.polygon([(220 + shift,154),(280 + shift,105),(340 + shift,105),(340 + shift,154)], fill="#243a50")
    draw.polygon([(352 + shift,105),(416 + shift,105),(470 + shift,154),(352 + shift,154)], fill="#243a50")
    draw.rectangle((516 + shift,185,550 + shift,215),fill="#fff6b0")
    return image


first = car("#e7e7e5")
first.save(OUT / "white_car_reference.png")
ImageEnhance.Brightness(car("#e7e7e5", 8)).enhance(0.82).save(OUT / "white_car_query.png")
car("#9f2934").save(OUT / "red_car.png")
for identity in ("white_car", "red_car", "blue_car"):
    (DATASET / identity).mkdir(parents=True, exist_ok=True)
car("#e7e7e5").save(DATASET / "white_car" / "reference.png")
ImageEnhance.Brightness(car("#e7e7e5", 8)).enhance(0.82).save(DATASET / "white_car" / "query.png")
car("#9f2934").save(DATASET / "red_car" / "reference.png")
ImageEnhance.Brightness(car("#9f2934", -7)).enhance(1.12).save(DATASET / "red_car" / "query.png")
car("#294f9f").save(DATASET / "blue_car" / "reference.png")
ImageEnhance.Brightness(car("#294f9f", 5)).enhance(0.9).save(DATASET / "blue_car" / "query.png")
print(f"Демонстрационные изображения созданы: {OUT}")
print(f"Демонстрационный датасет создан: {DATASET}")
