import numpy as np
from sklearn.metrics import confusion_matrix
# import scikit-image as skimage
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central
from skimage.measure import moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from collections import Counter
import skimage.morphology as morph

class Train():
    def __init__(self):
        self.train_data=['a.bmp','d.bmp','f.bmp','h.bmp',
                        'k.bmp','m.bmp','n.bmp','o.bmp',
                        'p.bmp','q.bmp','r.bmp','s.bmp',
                        'u.bmp','w.bmp','x.bmp','z.bmp']
        self.mean_list=[]
        self.std_var_list=[]
        # self.local_mean_list=[]
        # self.local_var_list=[]
        self.overall_tags=None
        self.local_tags=None
        self.local_original_features=None
        self.local_normalized_features=None
        self.overall_features=None
        self.overall_normalized_features=None
        self.predicted=[]
        self.indexer=0
    
    # using a binary search to find threshold
    def find_threshold_1(self, hist):
        threshold=0.0
        peak=np.where(hist[0]==np.max(hist[0]))[0][0]
        peak_2 = np.where(hist[0] == max(np.max(hist[0][:peak]), np.max(hist[0][peak+1:])))[0][0]
        if peak >= peak_2:
            valley = np.where(hist[0] == np.min(hist[0][peak_2:peak]))[0][0]
        else:
            valley = np.where(hist[0] == np.min(hist[0][peak:peak_2]))[0][0]

        th = hist[1][valley]
        return th

    # Triangle algorithm
    def find_threshold_2(self, hist):
        th = 0.0
        b_max = np.max(hist[0])
        index_max = np.where(hist[0] == b_max)[0][0]
        index_min = 0
        if len(np.where(hist[0] == 0)[0]):
            index_min = np.where(hist[0] == 0)[0][0]
        b_min = hist[0][index_min]

        x_max = hist[1][index_max]
        x_min = hist[1][index_min]

        A = b_max - b_min
        B = x_min - x_max
        C = b_min * (x_max - x_min) - x_min * (b_max - b_min)

        d = 0
        b_0 = index_min
        for i in range(index_min, index_max):
            distance_to_line = abs(A * hist[1][i] + B * hist[0][i] + C)
            if d < distance_to_line:
                d = distance_to_line
                b_0 = i

        th = hist[1][b_0]

        return th


    def feature_extraction(self, filename, mode, tag=None, binary_threshold=200, r_lower_threshold=10, c_lower_threshold=12, r_upper_threshold=80, c_upper_threshold=85, plot=False):
        img=io.imread(filename)
        # print(img.shape)
        # binarlize by thresholding
        hist=exposure.histogram(img)
        binary_threshold=self.find_threshold_2(hist)
        img_binary=(img<binary_threshold).astype(np.double)
        # extracting characters and their features

        # if using enhancement3, unblock following lines
        # # and block the next img_label definition line
        # img_morph=morph.binary_dilation(img_binary)
        # img_label=label(img_morph, background=0, connectivity=1)

        img_label=label(img_binary, background=0)
        regions=regionprops(img_label)



        # show the original image
        if plot==True:
            io.imshow(img)
            plt.title('original image')
            io.show()

            # image histogram
            hist=exposure.histogram(img)
            # if using find-threshold_1 as enhancement, unblock this following line
            # binary_threshold=self.find_threshold_1(hist)

            # if using find_threshold_2 as enhancement, unblock this following line
            binary_threshold=self.find_threshold_2(hist)

            # visualize the histogram as following
            plt.bar(hist[1], hist[0])
            plt.title("histogram")
            plt.show()
        
            # display binary image
            io.imshow(img_binary)
            plt.title("binary image")
            io.show()

            io.imshow(img_label)
            plt.title("labeled image")
            io.show()

            train_cache=Train()
            train_cache.feature_collection()
            train_cache.normalization(train_cache.overall_features)
            # print(train_cache.overall_normalized_features)
            train_cache.recognition(train_cache.overall_tags, train_cache.overall_normalized_features, train_cache.overall_normalized_features)
        
        # displaying component bounding boxes
        
        if plot==True:
            io.imshow(img_binary)
        ax=plt.gca()
        Features=[]
        if mode=="train":
            tags=[]
        elif mode=="test":
            pass
        index=0
        # print("region is: ",type(regions))
        # print(len(regions))
        for props in regions:
            minr, minc, maxr, maxc=props.bbox
            if plot==True:
                ax.add_patch(Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor="red", linewidth=1))
            roi=img_binary[minr:maxr, minc:maxc]

            # computing Hu moments and removing small components
            m=moments(roi)
            cr=m[0, 1]/m[0, 0]
            cc=m[1, 0]/m[0, 0]
            mu=moments_central(roi, cr, cc)
            nu=moments_normalized(mu)
            hu=moments_hu(nu)
            if maxr-minr<=r_lower_threshold or maxc-minc<=c_lower_threshold or maxr-minr>=r_upper_threshold or maxc-minc>=c_upper_threshold:
                continue
            else:
                Features.append(hu)
                if mode=="train":
                    tags.append(tag)
                elif mode=="test":
                    pass
            if plot==True:
                train_labeling=Train()
                train_labeling.local_feature_collection(filename)
                train_labeling.local_normalization(train_labeling.local_original_features, train_cache.mean_list, train_cache.std_var_list)
                train_labeling.recognition(train_cache.overall_tags, train_labeling.local_normalized_features, train_cache.overall_normalized_features)
                if index>=len(train_labeling.predicted):
                    break
                predicted_character=train_labeling.predicted[index]
                plt.text(maxc, minr, predicted_character, bbox=dict(facecolor='red', alpha=0.5))
            index+=1

        ax.set_title("bounding boxes")
        if plot==True:
            io.show()
        if mode=="train":
            return [np.array(tags),np.array(Features)]
        elif mode=="test":
            return Features

    def feature_collection(self):
        all_tags=list(self.feature_extraction(self.train_data[0], mode="train", tag=self.train_data[0][0])[0])
        all_features=list(self.feature_extraction(self.train_data[0], mode="train", tag=self.train_data[0][0])[1])
        for item in self.train_data[1:]:
            all_tags.extend(self.feature_extraction(item, mode="train", tag=item[0])[0])
            all_features.extend(self.feature_extraction(item, mode="train", tag=item[0])[1])
        self.overall_tags=all_tags
        self.overall_features=all_features

        return [self.overall_tags, self.overall_features]
    
    def local_feature_collection(self, filename):
        tags=list(self.feature_extraction(filename, mode="train", tag=filename[0])[0])
        features=list(self.feature_extraction(filename, mode="train", tag=filename[0])[1])
        self.indexer=len(features)
        self.local_original_features=features
        self.local_tags=tags

    def normalization(self, feature_list):
        result=[]
        for i in range(len(feature_list)):
            result.append([])
        for i in range(7):
            all=[]
            for item in feature_list:
                all.append(item[i])
            cache=[]
            mean=np.mean(all)
            self.mean_list.append(mean)
            std_var=np.std(all)
            self.std_var_list.append(std_var)
            for item in feature_list:
                cache.append((item[i]-mean/std_var))
            for j in range(len(result)):
                result[j].append(cache[j])
        # print(result[:10])
        self.overall_normalized_features=result
    
    def local_normalization(self, feature_list, mean_list, std_var_list):
        result=[]
        for i in range(len(feature_list)):
            result.append([])
        for i in range(7):
            cache=[]
            for item in feature_list:
                # print(len(mean_list))
                # print(len(self.std_var_list))
                cache.append((item[i]-mean_list[i]/std_var_list[i]))
            for j in range(len(result)):
                result[j].append(cache[j])
        self.local_normalized_features=result

    def recognition(self, tags, local_features, overall_features, show_visualizedD=False, show_confusion_matrix=False):
        train_label=['a','d','f','h',
                    'k','m','n','o',
                    'p','q','r','s',
                    'u','w','x','z']
        D=cdist(local_features, overall_features)
        if show_visualizedD==True:
            io.imshow(D)
            plt.title('Distance Matrix')
            io.show()
        accurate_count=0
        for index in range(len(local_features)):
            D_index=list(np.argsort(D[index], axis=0))
            index_cache=D_index[1]
            self.predicted.append(tags[index_cache])
            if tags[D_index[1]]==tags[index]:
                accurate_count+=1
            else:
                pass
        if show_confusion_matrix==True:
            confM=confusion_matrix(tags, self.predicted, labels=train_label)
            io.imshow(confM)
            plt.title('Confusion Matrix')
            io.show()
        
        print(accurate_count/len(tags))
        return accurate_count/len(tags)

def main():
    train=Train()
    train.local_feature_collection("a.bmp")
    train.feature_collection()
    train.normalization(train.overall_features)
    train.recognition(train.overall_tags, train.overall_normalized_features, train.overall_normalized_features, show_visualizedD=True)
    train_a=Train()
    train_a.local_feature_collection('a.bmp')
    train_a.feature_collection()
    train_a.normalization(train_a.overall_features)
    train_a.local_normalization(train_a.local_original_features, train.mean_list, train.std_var_list)
    train_a.feature_extraction('test2.bmp', mode="train", plot=True)
    # train_a.recognition()

if __name__=="__main__":
    main()