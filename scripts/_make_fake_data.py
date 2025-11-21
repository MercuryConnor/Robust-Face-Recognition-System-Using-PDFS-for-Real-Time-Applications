"""Create a tiny synthetic ImageFolder dataset using plain PPM images.

This avoids external imaging dependencies when generating a few sample
images for smoke tests.
"""
import os, random

def make_ppm(path, w=128, h=128):
    pixels = bytearray(random.getrandbits(8) for _ in range(w*h*3))
    with open(path, 'wb') as f:
        f.write(b'P6\n%d %d\n255\n' % (w, h))
        f.write(pixels)

def main():
    base = 'data/processed'
    os.makedirs(base, exist_ok=True)
    for cls in ('a','b'):
        d = os.path.join(base, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            make_ppm(os.path.join(d, f'image_{i}.ppm'))
    print('Created synthetic dataset at data/processed with classes a,b')

if __name__ == '__main__':
    main()
