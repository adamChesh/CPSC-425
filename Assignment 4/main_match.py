import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=0.9)
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/basmati',
        ratio_thres=0.9, orient_agreement=20, scale_agreement=0.2)
    plt.title('MatchRANSAC')
    plt.imshow(im)

if __name__ == '__main__':
    main()
