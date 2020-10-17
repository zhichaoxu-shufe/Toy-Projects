import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central
from skimage.measure import moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train import Train

class Test():
    def __init__(self):
        self.test_data=['test1.bmp', 'test2.bmp']
        self.local_orignial_features=None
        self.local_normalized_features=None
        self.local_tags=[]
        self.predicted=[]

    def feature_collection_test(self, filename, binary_threshold=200, r_lower_threshold=10, c_lower_threshold=12, r_upper_threshold=80, c_upper_threshold=85, plot=False):
        img=io.imread(filename)
        img_binary=(img<binary_threshold).astype(np.double)
        img_label=label(img_binary, background=0)
        regions=regionprops(img_label)

        if plot==True:
            io.imshow(img)
            plt.title('original image')
            io.show()

            # image histogram
            hist=exposure.histogram(img)
            # visualize the histogram as following
            plt.bar(hist[1], hist[0])
            plt.title('histogram')
            plt.show()

            # display binary image
            io.imshow(img_binary)
            plt.title("binary image")
            io.show()

            train_cache=Train()
            train_cache.feature_collection()
            train_cache.normalization(train_cache.overall_features)
            train_cache.recognition(train_cache.overall_tags, train_cache.overall_normalized_features, train_cache.overall_normalized_features)
        
        # displaying component bounding axes
        if plot==True:
            io.imshow(img_binary)
        ax=plt.gca()
        Features=[]
        index=0
        for props in regions:
            minr, minc, maxr, maxc=props.bbox
            if plot==True:
                ax.add_patch(Rectangle((minc, minr), maxc-minc,maxr-minr, fill=False, edgecolor='red', linewidth=1))
            roi=img_binary[minr:minc, minc:maxc]

            # computing Hu moments and removing small components
            m=moments(roi)
            cr=m[0, 1]/m[0, 0]
            cc=m[1, 0]/m[0, 0]
            mu=moments_central(roi, cr, cc)
            nu=moments_normalized(mu)
            hu=moments_hu(nu)
            if plot==True:
                train_labeling=Train()
                train_labeling.local_feature_collection(filename)
                train_labeling.local_normalization(train_labeling.local_original_features, train_cache.mean_list, train_cache.std_var_list)
                train_labeling.recognition(train_cache.overall_tags, train_labeling.local_normalized_features, train_cache.overall_normalized_features)
                if index>=len(train_labeling.predicted):
                    break
                predicted_character=train_labeling.predicted[index]
                plt.text(maxc, minr, predicted_character, bbox=dict(facecolor='red', alpha=0.5))
            if maxr-minr<=r_lower_threshold or maxc-minc<=c_lower_threshold or maxr-minr>=r_upper_threshold or maxc-minc>=c_upper_threshold:
                pass
            else:
                Features.append(hu)
            index+=1
        ax.set_title("bounding boxes")
        if plot==True:
            io.show()
        self.local_orignial_features=Features
        return Features
    
    def normalization(self, overall_features):
        train=Train()
        all_features=train.feature_collection()[1]
        train.normalization(all_features)
        
        result=[]
        for i in range(len(self.local_orignial_features)):
            result.append([])
        for i in range(7):
            cache=[]
            for item in self.local_orignial_features:
                cache.append((item[i]-train.mean_list[i])/train.std_var_list[i])
            for j in range(len(result)):
                result[j].append(cache[j])
        self.local_normalized_features=result
        return self.local_normalized_features

    # def recognition(self, filename, show_visualizedD=False):
    #     train=Train()
    #     train.feature_collection()
    #     train.normalization(train.overall_features)
    #     D=cdist(self.local_normalized_features, train.overall_normalized_features)
    #     if show_visualizedD==True:
    #         io.imshow(D)
    #         plt.title('Distance Matrix')
    #         io.show()
        # closest=[]
        # for index in range(len(features)):
        #     D_index=list(np.argsort(D[index], axis=0))
        #     closest.append(D_index[1])
        # ax=plt.gca()
        # index=0
        # for props in regions:
        #     minr, minc, maxr, maxc=props.bbox
        #     if show_visualizedD==True:
        #         ax.add_patch(Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=1))
        #     roi=img_binary[minr:maxr, minc:maxc]

        #     # computing hu moments and removing small components
        #     m=moments(roi)
        #     cr=m[0, 1]/m[0, 0]
        #     cc=m[1, 0]/m[0, 0]
        #     mu=moments_central(roi, cr, cc)
        #     nu=moments_normalized(mu)
        #     hu=moments_hu(nu)
        #     if show_visualizedD==True:
        #         closest=[]
        #         for index in range(len(self.local_normalized_features)):
        #             D_index=list(np.argsort(D[index], axis=0))
        #             closest.append(D[index[1]])


def main():
    test=Test()
    features=test.feature_collection_test('test1.bmp')

    test_1=Test()
    test_1.feature_collection_test('test1.bmp', plot=True)

    # test=Test()
    # features=test.feature_collection_test()
    # test.normalization(features)
    # test.recognition(test.normalized_features, show_visualizedD=False)
    # print(len(test.normalized_features))
    # train=Train()
    # all_tags=train.feature_collection()[0]
    # closest=test.recognition(test.normalized_features, show_visualizedD=False)
    # corresponding=[]
    # for i in closest:
    #     corresponding.append(all_tags[i])
    # for i in range(1, 9):
    #     print(corresponding[i*10-10: i*10])

    # print(len(features))
    # print(features[0])
    # normalized_features=test.normalization(features)
    # print(normalized_features[:3])


if __name__=="__main__":
    main()