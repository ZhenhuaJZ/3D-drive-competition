import pandas as pd
import os
from tqdm import tqdm

pts_path = './TrainSet/pts'
all_data_path = './TrainSet/all_data'
sampled_path = './TrainSet/samp_num_1w_5'
eda_path = './EDA'
eda_file_path = os.path.join(eda_path, 'eda_r5.txt')
cate_0_thresh = 5 #keep cate_0/cate_1_7 < 100's frames
cate_0_point_rate = 1 #keep cate_0 is <cate_0_point_rate> bigger than cate_1_7
num_aft_samp = 9000

# cate_0_thresh is dont care(category = 0) rate in one frame
# the bigger cate_0_thresh the more cate_0 in one frame (normal xxx)
def cate_0_in_one_frame_rate(file_path):
    eda_stage_1 = []
    only_cat_0_frame = []
    dir = os.listdir(file_path)
    for f in tqdm(dir):
        df = pd.read_csv(os.path.join(file_path , f))
        df = df['c'].value_counts()
        cate_0 = df.iloc[0] #find the max category appear frequency
        cate_1_7 = df.iloc[1:].sum() #find the rest category appear frequency
        
        if cate_1_7 == 0 :
            # print("NO OTHER CAT IN THIS FRAME")
       	    only_cat_0_frame.append(f)

        if (cate_0/cate_1_7) < cate_0_thresh:
            eda_stage_1.append(f)

    return eda_stage_1, only_cat_0_frame

def balance_each_frame(eda_path, file_path):

    balanced_frame = [] #origin frame  with good balanced category
    #READ EDA file save with coresponding cate_0_thresh
    with open(os.path.join(eda_path, "eda_r{}.txt".format(cate_0_thresh))) as f:
        txt = f.readlines()
        txt = list(map(lambda s: s.strip(), txt))

    for file in txt:
        _df = pd.read_csv(os.path.join(file_path, file))
        df = _df['c'].value_counts()
        cate_0 = df.iloc[0] #find the max category appear frequency
        cate_1_7 = df.iloc[1:].sum() #find the rest category appear frequency

        keep_cate_0 = int(cate_0_point_rate * cate_1_7) #ramdom delete cate_0_del from cate_0
        drop_frq = 1- (keep_cate_0/cate_0)

        if drop_frq <= 0:

            print("Good qulity frame")
            balanced_frame.append(file)

        else:
            _df = _df.drop(_df.loc[_df['c'] == 0].sample(frac = drop_frq).index) # random drop 0 and make new df 
            _df = _df['c'].value_counts()
            cate_0 = _df.iloc[0] #find the max category appear frequency
            cate_1_7 = _df.iloc[1:].sum() #find the rest category appear frequency
            

def save_eda_to_txt(eda_stage_1, only_cat_0_frame):

    for file in eda_stage_1:
        with open("EDA/eda_r{}".format(cate_0_thresh) + ".txt", 'a') as f :
            f.write((file) + "\n")

    for file in only_cat_0_frame:
    	with open("EDA/eda_only_cate_0" + ".txt", 'a') as f :
            f.write((file) + "\n")

def frame_subsamp(eda_file_path, file_path, to_path):
	print('into file')
	print(eda_file_path)
	file_name_list = []
	counter = 0
	with open(eda_file_path) as f:
		file_name_list = f.readlines()
		# file_name_list.append(file_name)
		file_name_list = list(map(lambda s: s.strip(), file_name_list))

	for file_name in tqdm(file_name_list):
		# print(file_name)
		# print(file_name_list[1])
		df = pd.read_csv(os.path.join(file_path, file_name))
		cate_num_1_7 = df['c'].value_counts().iloc[1:].sum()
		if cate_num_1_7 > num_aft_samp:
			df[df['c'] > 0].sample(9000).to_csv(os.path.join(to_path, file_name), index = False)

		if cate_num_1_7 < num_aft_samp:
			cate_num_2_7 = df['c'].value_counts().iloc[2:].sum()
			cate_2_7 = df[(df['c'] != df['c'].value_counts().index[1]) & (df['c'] != df['c'].value_counts().index[0])]
			# print(df['c'].value_counts())
			# print((num_aft_samp-cate_num_2_7)/2)
			if int(df['c'].value_counts().iloc[1]) > (num_aft_samp-cate_num_2_7)/2:
				cate_0 = df[df['c'] == df['c'].value_counts().index[0]].sample(int((num_aft_samp-cate_num_2_7+1)/2))
				cate_1 = df[df['c'] == df['c'].value_counts().index[1]].sample(int((num_aft_samp-cate_num_2_7+1)/2))
			else:
				cate_0 = df[df['c'] == df['c'].value_counts().index[0]].sample(num_aft_samp-cate_num_1_7)
				cate_1 = df[df['c'] == df['c'].value_counts().index[1]]
			# print(len(cate_0), len(cate_1))
			cate_0_7 = pd.concat([cate_2_7, cate_0, cate_1]).reset_index(drop = True)
			# print(len(cate_0_7))
			# print(cate_0_7['c'].value_counts())
			cate_0_7.to_csv(os.path.join(to_path, file_name), index = False)
		
def find_max_z(eda_file_path, file_path, to_path):
    print('into file')
    print(eda_file_path)
    file_name_list = []
    counter = 0
    maxz1 = 0
    maxz2 = 0
    maxz3 = 0
    maxz4 = 0
    maxz5 = 0
    maxz6 = 0
    maxz7 = 0
    with open(eda_file_path) as f:
        file_name_list = f.readlines()
        # file_name_list.append(file_name)
        file_name_list = list(map(lambda s: s.strip(), file_name_list))

    for file_name in tqdm(file_name_list):
        # print(file_name)
        # print(file_name_list[1])
        pts = pd.read_csv(os.path.join(file_path, file_name))
        x = 60
        print("x max", pts['x'].max())
        print("x min", pts['x'].min())
        # print('*****')
        # print("x number of between -{} to {} meters: ".format(x,x), len(pts[(pts['x'] > -x) & (pts['x'] < x)]))
        in_list_value = pts[(pts['x'] > -x) & (pts['x'] < x)]['c'].value_counts()
        # print("label between -{} to {} meters: \n".format(x,x), in_list_value)
        # print('*****')
        # print("x number of outside -{} meters: ".format(x), len(pts[(pts['x'] <= -x) | (pts['x'] >= x)]))
        out_list_value = pts[(pts['x'] <= -x) | (pts['x'] >= x)]['c'].value_counts()
        # print("label outside -{} meters: \n".format(x,x), list_value[list_value.index > 0])

        print("percentiage outiside {} meters".format(x), out_list_value/in_list_value)

        # print("x total number: ", len(pts['x']))
        # print(sum(pts.iloc[:,2] > 8)/float(len(pts.iloc[:,2])))
        print('*********************************\n')
        # z7 = df[df['c']==7]['z'].max()
        # z1 = df[df['c']==1]['z'].max()
        # z2 = df[df['c']==2]['z'].max()
        # z3 = df[df['c']==3]['z'].max()
        # z4 = df[df['c']==4]['z'].max()
        # z5 = df[df['c']==5]['z'].max()
        # z6 = df[df['c']==6]['z'].max()
        # if z1 > maxz1:
        #     maxz1 = z1
        #     print("1: ",maxz1)
        # if z2 > maxz2:
        #     maxz2 = z2
        #     print("2: ",maxz2)
        # if z3 > maxz3:
        #     maxz3 = z3
        #     print("3: ",maxz3)
        # if z4 > maxz4:
        #     maxz4 = z4
        #     print("4: ",maxz4)
        # if z5 > maxz5:
        #     maxz5 = z5
        #     print("5: ",maxz5)
        # if z6 > maxz6:
        #     maxz6 = z6
        #     print("6: ",maxz6)
        # if z7 > maxz7:
        #     maxz7 = z7
    #         print("7: ",maxz7)
    # print("maxz1: ",maxz1)
    # print("maxz2: ",maxz2)
    # print("maxz3: ",maxz3)
    # print("maxz4: ",maxz4)
    # print("maxz5: ",maxz5)
    # print("maxz6: ", maxz6)
    # print("maxz7: ",maxz7)


# frame_subsamp(eda_file_path, all_data_path, sampled_path)
# find_max_z(eda_file_path, all_data_path, sampled_path)
eda_stage_1, only_cat_0_frame = cate_0_in_one_frame_rate(all_data_path)
save_eda_to_txt(eda_stage_1, only_cat_0_frame)
# balance_each_frame(eda_path, all_data_path)
