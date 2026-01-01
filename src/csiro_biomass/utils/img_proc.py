"""image processing."""


def convert_to_8_tile(img):
    img1 = img.crop((0, 0, 500, 500))
    img2 = img.crop((500, 0, 1000, 500))
    img3 = img.crop((1000, 0, 1500, 500))
    img4 = img.crop((1500, 0, 2000, 500))

    img5 = img.crop((0, 500, 500, 1000))
    img6 = img.crop((500, 500, 1000, 1000))
    img7 = img.crop((1000, 500, 1500, 1000))
    img8 = img.crop((1500, 500, 2000, 1000))

    img_resized = img.resize((500, 500))

    img_tiles = [img_resized, img1, img2, img3, img4, img5, img6, img7, img8]

    return img_tiles
