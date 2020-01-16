from util import build_codebook

FRAME_DIR = './frames/'
SIFT_DIR = './sift/'
NUM_WORDS = 500
NUM_FRAMES = 100
NUM_DESC = 2000
SAMPLE_STEP = 5
M = 5

if __name__ == '__main__':
    build_codebook(NUM_WORDS, NUM_FRAMES)