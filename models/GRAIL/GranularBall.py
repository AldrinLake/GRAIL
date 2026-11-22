from collections import Counter
from sklearn.cluster._kmeans import k_means
import numpy as np
import random
import torch
import copy
import warnings
from sklearn.neighbors import LocalOutlierFactor
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

_seed = 1993
random.seed(_seed)

class GBList:
    def __init__(self, smp_feat, smp_label, task_id):
        data = data_processing(smp_feat, smp_label)
        self.data = data
        self.alldata = data
        self.task_id = task_id
        self.granular_balls = [GranularBall(self.data, task_id)]
        self.dict_granular_balls = {}
        self.max_first_task_size = 0


    def init_granular_balls_dict(self, min_purity = 0.51, max_purity = 1.0, min_sample=1):
        """
            Function function: to obtain the particle partition within a certain purity range
            Input: minimum purity threshold, maximum purity threshold, minimum number of points in the process of pellet division
        """
        for i in range(int((max_purity - min_purity) * 100) + 1):
            purity = i / 100 + min_purity
            self.init_granular_balls(purity, min_sample)
            self.dict_granular_balls[purity] = self.granular_balls.copy()

    def init_granular_balls(self, purity=1.0, min_sample=1):  # Set the purity threshold to 1.0
        """
            Function function: calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        ll = len(self.granular_balls)  # Record total number of balls
        i = 0  # Counter representing balls stable enough to stop splitting
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                    granular_ballsitem.getBoundaryData()
                break
        
            
    def init_granular_balls2(self, purity=1.0, max_sample=20):  # Set the purity threshold to 1.0
        """
            Function function: calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        ll = len(self.granular_balls)  # Record total number of balls
        i = 0  # Counter representing balls stable enough to stop splitting
        while True:
            if self.granular_balls[i].purity < purity or self.granular_balls[i].num > max_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    # print("出现分裂后没分的情况")
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                    granular_ballsitem.getBoundaryData()
                break
        
        # Identify and remove outlier balls based on intra-class distances
        lables_pool = np.unique(list(map(lambda x: x.label, self.granular_balls)))
        T_ball = []
        for label in lables_pool:
            ball_centers = np.array(list(map(lambda x: x.center, filter(lambda x:x.label==label, self.granular_balls))))
            balls = np.array(list(map(lambda x: x, filter(lambda x:x.label==label, self.granular_balls))))
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.5)  # proportion of outliers
            y_pred = lof.fit_predict(ball_centers)  # -1: outlier, 1: inlier
            outlier_indices = np.where(y_pred == -1)[0] # indices of outliers
            # print(outlier_indices)
            for idx, ball in enumerate(balls):
                if idx not in outlier_indices:
                    T_ball.append(ball)
        self.granular_balls = T_ball.copy()

    def init_granular_balls3(self, purity=1.0, min_sample=1):  # Set the purity threshold to 1.0
        """
            Function function: calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        ll = len(self.granular_balls)  # Record total number of balls
        i = 0  # Counter representing balls stable enough to stop splitting
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                    # granular_ballsitem.getBoundaryData()
                break
        lables_pool = np.unique(list(map(lambda x: x.label, self.granular_balls)))
        T_ball = []
        for label in lables_pool:
            average_size = np.mean(list(map(lambda x: len(x.data), filter(lambda x: x.label==label, self.granular_balls))))
            T_ball.extend(list(map(lambda x: x, filter(lambda x: len(x.data) >= average_size and x.label==label, self.granular_balls))))
        self.granular_balls = T_ball.copy()
    
    
    def init_granular_balls4(self, purity=1.0, min_sample=1, remain_ball_num=1):  # Set the purity threshold to 1.0
        """
            Function function: calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        ll = len(self.granular_balls)  # Record total number of balls
        i = 0  # Counter representing balls stable enough to stop splitting
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                break

    def resplit(self):
        max_iteration = 1000
        while True:
        # for i in range(max_iteration):
            ball_centers = self.get_center()
            ball_labels = self.get_label()
            ball_radius = self.get_radius()
            distances = cdist(ball_centers, ball_centers, metric='euclidean')
            same_label_mask =  (ball_labels[:, np.newaxis] == ball_labels)
            pail_ridius_sum = ball_radius[:, np.newaxis] + ball_radius
            overlap_dif_cls_balls = (distances < pail_ridius_sum) & ~same_label_mask    # 判断两个异类球是否重叠
            has_true = np.any(overlap_dif_cls_balls, axis=1)  # 判断每一行是否包含 True
            true_row_indices = np.where(has_true)[0]# 获取包含 True 的行的索引
            # print("{}:{}".format(i, true_row_indices))
            if len(true_row_indices) == 0:
                break
            wait_pop = []
            for ball_idx in true_row_indices:
                # If a ball has only two samples, splitting may produce single-sample balls; split_clustering returns []
                split_clusters = self.granular_balls[ball_idx].split_clustering()
                if len(split_clusters) > 1:
                    self.granular_balls[ball_idx] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                elif len(split_clusters) == 1:
                    self.granular_balls[ball_idx] = split_clusters[0]
                else:
                    wait_pop.append(ball_idx)   # If empty, original ball produced single-sample balls; remove to avoid infinite loop
            # print("{}:{}".format(i, wait_pop))
            self.granular_balls = copy.deepcopy([self.granular_balls[i] for i in range(len(self.granular_balls)) if i not in wait_pop]) 
            for ball in self.granular_balls:
                ball.get_radius()

    def ball_fusion(self):
        """
        Note: The ball radius should be the maximum distance, not the mean. Otherwise:
        1) A large ball can keep merging nearest same-class small balls with little radius change, keeping purity at 1 and ignoring constraints;
        2) Eventually each class may end up with only one huge ball.
        """
        max_iteration = 1000
        while True:
        # for j in range(max_iteration):
            ball_centers = self.get_center()
            ball_labels = self.get_label()
            distances = cdist(ball_centers, ball_centers, metric='euclidean')
            same_label_mask =  (ball_labels[:, np.newaxis] == ball_labels)
            filtered_dist_matrix = np.full_like(distances, np.inf)
            filtered_dist_matrix[same_label_mask] = distances[same_label_mask]
            np.fill_diagonal(filtered_dist_matrix, np.inf)
            nearest_indices = np.argmin(filtered_dist_matrix, axis=1)
            ball1_before_fusion = []
            ball2_before_fusion = []
            ball3_not_fusion = []
            new_balls = []
            balls_idx = list(range(len(self.granular_balls)))
            for i in range(len(nearest_indices)):
                if i not in balls_idx:  # This ball has already been considered/merged
                    continue
                if nearest_indices[i] not in balls_idx:  # Nearest same-class neighbor already merged
                    ball3_not_fusion.append(self.granular_balls[i])
                    balls_idx.remove(i)
                    continue
                if np.all(np.isinf(filtered_dist_matrix[i])):
                    ball3_not_fusion.append(self.granular_balls[i])
                    balls_idx.remove(i)
                    continue
                ball1_idx = i
                ball2_idx = nearest_indices[i]
                
                new_ball = GranularBall(np.concatenate((self.granular_balls[ball1_idx].data, self.granular_balls[ball2_idx].data), axis=0), self.task_id)
                new_ball.get_radius()

                new_balls.append(new_ball)
                ball1_before_fusion.append(self.granular_balls[ball1_idx])
                ball2_before_fusion.append(self.granular_balls[ball2_idx])

                balls_idx.remove(ball1_idx)
                balls_idx.remove(ball2_idx)

            new_ball_centers = np.array(list(map(lambda x: x.center, new_balls)))
            new_ball_radius = np.array(list(map(lambda x: x.radius, new_balls)))
            if len(new_ball_centers) < 1:
                break
            dist = cdist(new_ball_centers, self.alldata[:, :-2], metric='euclidean')
            indices_within_radius = [np.where(dist[i] <= new_ball_radius[i])[0] for i in range(len(new_ball_centers))]
            final_balls = []
            flag = True
            for i in range(len(new_balls)):
                if len(self.alldata[indices_within_radius[i], :]) >=2:
                    tmp_ball = GranularBall(self.alldata[indices_within_radius[i], :], self.task_id)
                    if tmp_ball.purity < 1 or new_balls[i].purity < 1:
                        final_balls.append(ball1_before_fusion[i])
                        final_balls.append(ball2_before_fusion[i])
                    else:
                        flag = False
                        final_balls.append(new_balls[i])
                else:
                    flag = False
                    final_balls.append(new_balls[i])
            final_balls.extend(ball3_not_fusion)
            self.granular_balls = copy.deepcopy(final_balls)
            if flag:
                break

    def ball_selection2(self):
        remain_ball_num = 5
        lables_pool = np.unique(list(map(lambda x: x.label, self.granular_balls)))
        T_ball = []
        # Keep the top-k largest balls per class
        for label in lables_pool:
            size_list = list(map(lambda x: len(x.data), filter(lambda x: x.label==label, self.granular_balls)))
            values_sorted = sorted(size_list, reverse=True)  # Descending
            if remain_ball_num < len(values_sorted):
                trshold_size = values_sorted[remain_ball_num-1]
            else:
                trshold_size = values_sorted[-1]
            T_ball.extend(list(map(lambda x: x, filter(lambda x: len(x.data) >= trshold_size and x.label==label, self.granular_balls))))
        self.granular_balls = T_ball.copy()


    def ball_selection(self, _max=5):
        lables_pool = np.unique(list(map(lambda x: x.label, self.granular_balls)))
        T_ball = []
        # Adaptive selection based on data distribution
        for label in lables_pool:
            sorted_size_list = sorted(self.get_data_size_with_class_filter(_class=label), reverse=True)
            size_list =list(set(self.get_data_size_with_class_filter(_class=label)))
            if len(size_list) >= 2:
                density = gaussian_kde( np.array(size_list))  # Estimate density function
                x = np.linspace(min(size_list), max(size_list), 1000)
                y = density(x)
                # Compute first and second derivatives; find point of maximum curvature
                dy = np.gradient(y, x)
                d2y = np.gradient(dy, x)
                threshold_index = np.argmax(d2y)
                trshold_size = x[threshold_index]

                greater_than_threshold = sorted_size_list > trshold_size
                # Use argmax on reversed array to find last index above threshold
                last_index = np.argmax(greater_than_threshold[::-1])

                # For later tasks, cap the number of selected balls per class by the first task's maximum
                if self.task_id > 0:
                    # print("idx:{},_max:{}".format(last_index,_max))
                    if last_index > _max:
                        trshold_size = sorted_size_list[_max]
                    # print("trshold_size:{}, max_first_task_size:{}, sorted_size_list[_max]:{}".format(trshold_size, _max, sorted_size_list[_max]))

                # Select significantly larger balls
                Temp = list(map(lambda x: x, filter(lambda x: len(x.data) > trshold_size and x.label==label, self.granular_balls)))
                if len(Temp) == 0:
                    Temp = list(map(lambda x: x, filter(lambda x: len(x.data) >= sorted_size_list[0] and x.label==label, self.granular_balls)))
                T_ball.extend(Temp)
                if self.task_id == 0:
                    self.max_first_task_size = len(Temp) if len(Temp) > self.max_first_task_size else self.max_first_task_size
                    # print("max_first_task_size:{}, len(Temp):{}".format(self.max_first_task_size, len(Temp)))
            else:
                T_ball.extend(list(map(lambda x: x, filter(lambda x: x.label==label, self.granular_balls))))
        self.granular_balls = copy.deepcopy(T_ball)

    
    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))
    
    def get_data_size_with_class_filter(self, _class):
        return list(map(lambda x: len(x.data), filter(lambda x: x.label == _class, self.granular_balls)))

    def get_data_size_with_task_filter(self, _task):
        return list(map(lambda x: len(x.data), filter(lambda x: x.task_id == _task, self.granular_balls)))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))
    
    def get_purity_with_class_filter(self, _class):
        return list(map(lambda x: x.purity, filter(lambda x: x.label == _class, self.granular_balls)))
    

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))
    
    def get_center_with_class_filter(self, _cls):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, filter(lambda x: x.label == _cls, self.granular_balls))))
    
    def get_balls_with_class_filter(self, _cls):
        """
        :return: balls with specified class.
        """
        return np.array(list(map(lambda x: x, filter(lambda x: x.label == _cls, self.granular_balls))))
    
    def get_one_center_label_radius_per_class(self, start_class, end_class):
        """
        :return: a radom ball center of each class.
        """
        final_centers, labels, radius_var = [], [], []
        for _clss in range(start_class, end_class):
            balls = self.get_balls_with_class_filter(_clss)
            idx = list(range(len(balls)))
            np.random.shuffle(idx)
            final_centers.append(balls[idx[0]].center)
            labels.append(_clss)
            radius_var.append(balls[idx[0]].radius_var)
        return copy.deepcopy(np.array(final_centers)), np.array(labels), np.array(radius_var)

    def get_one_center_label_radius_enlarged_per_class(self, start_class, end_class, cur_task, blur_factor=0.03):
        """
        :return: a radom ball center of each class.
        """
        final_centers, labels, radius_var = [], [], []
        for _clss in range(start_class, end_class):
            balls = self.get_balls_with_class_filter(_clss)
            idx = list(range(len(balls)))
            np.random.shuffle(idx)
            final_centers.append(balls[idx[0]].center)
            labels.append(_clss)
            # w_r = 1+ np.log(1+(cur_task - balls[idx[0]].task_id)) / np.log(20) # 效果不好：{10, 20}
            w_r = 1+ (cur_task - balls[idx[0]].task_id) * blur_factor
            radius_var.append(balls[idx[0]].radius_var * w_r)
        return copy.deepcopy(np.array(final_centers)), np.array(labels), np.array(radius_var)

    def get_radius(self):
        """
        :return: the radius of each ball.
        """
        return np.array(list(map(lambda x: x.radius, self.granular_balls)))

    def get_radius_var(self):
        """
        :return: the radius_var of each ball.
        """
        return np.array(list(map(lambda x: x.radius_var, self.granular_balls)))
    
    def get_fea_aug_by_center(self):
        """
        :return: the feature augmented by center and radius_var
        """
        return np.array(list(map(lambda x: x.center + np.random.normal(0, 1, 512) * x.radius_var, self.granular_balls)))

    
    def get_center_with_size_filter(self, size_T):
        """
        :return: the center of each ball that larger than a size.
        """
        return np.array(list(map(lambda x: x.center, filter(lambda x: len(x.data) >= size_T, self.granular_balls))))
    
    def get_center_and_label_of_ball_large_than_average_size_of_class(self):
        """
        :return: the center and label of balls whose size larger than the average value of the class
        """
        lables_pool = np.unique(list(map(lambda x: x.label, self.granular_balls)))
        # print("labels pool:{}".format(lables_pool))
        centers = []
        labels = []
        for l in lables_pool:
            average_size = np.mean(list(map(lambda x: len(x.data), filter(lambda x: x.label==l, self.granular_balls))))
            # print(average_size)
            centers.append(list(map(lambda x: x.center, filter(lambda x: len(x.data) >= average_size and x.label==l, self.granular_balls))))
            labels.append(list(map(lambda x: x.label, filter(lambda x: len(x.data) >= average_size and x.label==l, self.granular_balls))))
            # print("class:{}, prot_num:{}".format(l, len(centers)))

        centers = np.concatenate(centers, axis=0)  # 合并为 (12, 512)
        labels = np.concatenate(labels, axis=0)   
        # print(centers.shape)
        # print(labels.shape)
        return centers, labels
        

    def get_label_with_size_filter(self, size_T):
        """
        :return: the label of each ballthat larger than a size.
        """
        return np.array(list(map(lambda x: x.label, filter(lambda x: len(x.data) >= size_T, self.granular_balls))))
    
    def get_label(self):
        """
        :return: the label of each ball
        """
        return np.array(list(map(lambda x: x.label, self.granular_balls)))


    def get_r(self):
        """
        :return: return radius r
        """
        return np.array(list(map(lambda x: x.radius, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def get_ball_num(self):
        """
        return: number of granular balls in the list
        """
        return len(self.granular_balls)
    
    def merge_new_ball_list(self, gblist):
        for gb in gblist.granular_balls:
            self.granular_balls.append(gb)

    def del_ball(self, purty=0., num_data=0):
        # delete ball
        T_ball = []
        for ball in self.granular_balls:
            if ball.purity >= purty and ball.num >= num_data:
                T_ball.append(ball)
        self.granular_balls = T_ball.copy()
        self.data = self.get_data()

import numbers
import os
class GranularBall:
    def __init__(self, data, task_id):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape  # Number of rows, number of columns
        self.center = self.data_no_label.mean(0)  # According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.label, self.purity = self.__get_label_and_purity()  # The type and purity of the label to put back the pellet
        self.init_center = self.random_center()  # Get a random point in each tag
        self.label_num = len(set(data[:, -2]))
        self.boundaryData = None
        self.radius = None
        self.radius_var = None
        self.task_id = task_id

        if not isinstance(self.label, numbers.Number):
            print(self.label)
            raise SystemExit

    def random_center(self):
        """
            Function function: saving centroid
            Return: centroid of all generated clusters
        """
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        if len(set(self.data[:, -2]))>=2:    # If >=2 classes exist in this ball, choose one center per class
            for i in set(self.data[:, -2]):
                data_set = self.data_no_label[self.data[:, -2] == i, :]  # A label is equal to the set of all the points to label a point
                random_data = data_set[random.randrange(len(data_set)), :]  # A random point in the dataset
                center_array = np.append(center_array, [random_data], axis=0)  # Add to the line
        else:   # Single class: pick random centers
            idx = np.random.choice(self.num, size=min(2, self.num), replace=False)
            center_array = self.data_no_label[idx]
        return center_array

    def __get_label_and_purity(self):
        """
           Function function: calculate purity and label type
       """
        count = Counter(self.data[:, -2])  # Counter, put back the number of class tags
        label = int(max(count, key=count.get))  # Get the maximum number of tags
        purity = count[label] / self.num  # Purity obtained, percentage of tags
        return label, purity

    def get_radius(self):
        """
           Function function: calculate radius based on distance or variance
        """
        diffMat = np.tile(self.center, (self.num, 1)) - self.data_no_label
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        # self.radius = distances.sum(axis=0) / self.num
        self.radius = distances.max()
        self.radius_var = np.mean(np.var(self.data_no_label, axis=0))

    def plot_fea_var(self, save_path):
        """
        Function: plot variance pic
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        variances = np.var(self.data_no_label, axis=0)
        color = 'darkblue' 
        plt.figure(figsize=(14, 7))  # 增加图形尺寸
        bars = plt.bar(range(len(variances)), variances, color=color, alpha=1.0, width=1.0, edgecolor='black', linewidth=1.2, zorder=3)

        for bar in bars:
            # bar.set_edgecolor('black')  # 确保有黑色的边框
            # 添加阴影：在原柱子的基础上，绘制一个稍微向右下方偏移的灰色矩形
            shadow = plt.Rectangle((bar.get_x() + 0.02, bar.get_height() * 0.02), bar.get_width(), bar.get_height(), color='gray', alpha=0.3, zorder=2)
            plt.gca().add_patch(shadow)
        # 添加标题和标签
        plt.title('Variance of Features (Diagonal Elements of Covariance Matrix)', fontsize=16)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Variance', fontsize=12)
        # 设置y轴范围为0到0.4
        plt.ylim(0, 0.4)
        # 使x轴标签不重叠，旋转并设置间距
        plt.xticks(range(0, len(variances), 50), rotation=45, fontsize=10)
        # 显示网格线
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.savefig(save_path + "/label{}_size{}_{}".format(self.label, self.num, random.randint(100, 999)))



    def split_clustering2(self):
        """
           Function function: continue to divide the granule into several new granules
           Output: new pellet list
       """
        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=len(self.init_center))
        data_label = ClusterLists[1]  # Get a list of tags
        for i in range(len(self.init_center)):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) > 1:   # Ensure at least two samples per cluster
                Cluster = GranularBall(Cluster_data, self.task_id)
                Clusterings.append(Cluster)
        return Clusterings
    
    def split_clustering(self):
        """
           Single sample can also be treated as a ball
        """
        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=len(self.init_center))
        data_label = ClusterLists[1]  # Get a list of tags
        for i in range(len(self.init_center)):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) >= 1:   # A single sample can form a ball
                Cluster = GranularBall(Cluster_data, self.task_id)
                Clusterings.append(Cluster)
        return Clusterings



def data_processing(feature, label):
    """
    feature: Each row is a sample, feature dimension 512
    label: Each row is a single non-one-hot label
    Data normalization:
    1. Ensure numpy input when constructing granular balls
    2. Concatenate features and labels; penultimate column is label, last column is index
    """
    
    if isinstance(feature, torch.Tensor):
        if feature.requires_grad:
            feature = feature.detach()  # detach first to avoid gradient tracking
        feature = feature.cpu().numpy()
    elif isinstance(feature, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported data type: {type(feature)}")

    index = np.arange(feature.shape[0])
    label = label.reshape(-1, 1)  # (sample_num, 1)
    index = index.reshape(-1, 1)  # (sample_num, 1)

    concatenated_data = np.hstack((feature, label, index))   # (sample_num, 514)
    return concatenated_data


if __name__ == "__main__":
    data = np.load('models/GRAIL/data_CIFAR100_task_0.npy', allow_pickle=True)
    data_dict = data.item()  # Must use .item() to get the dict
    feature = data_dict["feature"].detach().numpy()
    label = data_dict["label"]

    granular_balls = GBList(feature, label, task_id=0)
    granular_balls.init_granular_balls(purity=1)  # initialization
    # print(granular_balls.get_data_size())
    # print("remain sample num: {}".format(np.sum(granular_balls.get_data_size())))
    # print("ball_num:{}".format(granular_balls.get_ball_num()))
    pass