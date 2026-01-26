import time

from PIL import Image, ImageDraw, ImageFont

import st7735

MESSAGE = "ST7735/160*128px/Python"

# Create ST7735 LCD display class.
disp = st7735.ST7735(
    port=0,
    cs=0,
    dc=24,
    rst=25,
    rotation=90,
    spi_speed_hz=8000000,
    width=128,
    height=160,
    offset_left=0,
)

# Initialize display.
disp.begin()

WIDTH = disp.width
HEIGHT = disp.height


img = Image.new('RGB', (WIDTH, HEIGHT), color=(0, 0, 0))

draw = ImageDraw.Draw(img)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)

x1, y1, x2, y2 = font.getbbox(MESSAGE)
size_x = x2 - x1
size_y = y2 - y1

text_x = 160
text_y = (128 - size_y) // 2

t_start = time.time()

while True:
    x = (time.time() - t_start) * 100
    x %= (size_x + 160)
    draw.rectangle((0,0,160,128), (0, 0, 0))
    draw.rectangle((0,0,40,40), (256,0,0))
    draw.rectangle((40,0,80,40), (0,256,0))
    draw.rectangle((80,0,120,40), (0,0,256))
    draw.text((int(text_x - x), text_y), MESSAGE, font=font, fill=(255, 255, 255))
    disp.display(img)