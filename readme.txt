Code Usage Manual

1.First of all, when we download the dataset of one tast, we should take 'zarr' to 'txt', to handle it conviniently. Please use 'readzarr.py' to do it!

2.When we get the 'txt' dataset, please notice the data's meaning. We should use the delta of the trajectory to control the robotic in our environment (pybullet), to get the joint information of the robot. Through our code, we can get the 'q, dotq, tau' information of the joint of the robotic, and we just use 'q' & 'dotq'
The code folder of the environment named "direct_data". In this folder, we can get the code named "visual-multi-time.py", please run this code to get the joint information. notice! We should change our data in "visual-multi-time.py" and change the name of output file in "block_pushing_multimodal.py".

3. Because of the environment, we should cancel the 1, 7,13 columns, this columns are zero and have no use. please use 'delete_column_txt.py' to do it.

4.When we get the joint data, we can use ''qdd.py" to calculate the dotdotq of the joint of robotic. Now, we have a 18 columns data file, include 6 columns of q, q columns of dq, 6 columns of ddq.

5.Then, we use the 18 columns data to calculate the touques of 6 joint. For this thing, we use matlab. We can into the touque_matlab folder to find a tau_calculate.m, and put our 18 columns data in, and then change ournumber of row of the data in code. Then run it to get the 6 columns of touques.

6.Then we should concatenate data. First, we use "combine_data.py" to concatenate the "q dq ddq" data and the trajectory data, than we have a 20 columns, include 18 columns of joint information and 2 columns trajectory data. After that, we  concatenate the 20 columns with the 6 columns of touques. And now, we have a complete dataset with 26 coluimns.

7.To make the training loss more relable, we should normailization our data, we can use "normalization.py" to do it.

8.And now, we have a complete dataset to train "dynamic with trajectory" neural network. We use "train_test_data.py" to divide our data into two parts, one for 80% for training, and one for 20% for testing.

9.Now, we use "LSTM_4_mask.py" to train the model

10. Then we have the model, and use "LSTM_4_test_mask.py" to test the model's effect.



