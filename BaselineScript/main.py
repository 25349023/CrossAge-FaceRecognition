from Evaluation import Matrix
from ModelFactory import FaceFeatureExtractor
import numpy as np
import warnings
from PIL import Image


def main():
    # Close the warning 
    # warnings.filterwarnings("ignore")

    inference_insightface()


def inference_insightface():
    print("==> test the insighface")
    img1_pathstr = 'Test/Aaron_Guiel_0001.jpg'
    img2_pathstr = 'Test/Aaron_Guiel_0002.jpg'
    img3_pathstr = 'Test/Abdullah_Ahmad_Badawi_0001.jpg'
    # Load model
    facerecognition = FaceFeatureExtractor.insightFace("mobilefacenet")

    # image processing 
    img1 = Image.open(img1_pathstr).resize((112, 112))
    img2 = Image.open(img2_pathstr).resize((112, 112))
    img3 = Image.open(img3_pathstr).resize((112, 112))

    # Get feature and compare 
    img1_feat = facerecognition.extract_feat(img1)
    img2_feat = facerecognition.extract_feat(img2)
    img3_feat = facerecognition.extract_feat(img3)

    # Check
    score12 = Matrix.get_cosine_similarity(img1_feat, img2_feat)[0, 0]
    score13 = Matrix.get_cosine_similarity(img1_feat, img3_feat)[0, 0]

    print(f'similarity score between 1 & 2:{score12}')
    print(f'similarity score between 1 & 3:{score13}')


if __name__ == '__main__':
    main()
