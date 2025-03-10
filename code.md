---
layout: default
title: Code
---

<h1 style="text-align: center;">Continue Updating...</h1>

<div style="margin-top: 20px; max-width: 800px; margin: 0 auto;">
    <h1 style="text-align: center;">Code of Pre-train Module</h1>
    
    <div style="text-align: center; margin: 20px 0;">
        <a href="https://github.com/zhangyurou/PI-t" class="github-link" style="font-size: 1.2em; text-decoration: none; color: #0366d6;">
            <img src="{{ '/assets/images/Github_mark.png' | relative_url }}" alt="GitHub Logo" style="width: 120px; vertical-align: middle; margin-right: 10px;">
            Visit our GitHub Repository
        </a>
    </div>

    <div style="text-align: justify;">
        <h2>Introduction</h2>
        <p>
            This is the official implementation of the paper "πₜ: Physics-Informed Plug-in Model for Robotic Trajectory Generation". Our code includes the implementation of the physics-informed neural network (PINN) for trajectory refinement and the dataset used in our experiments.
        </p>

        <h2>Installation</h2>
        <pre style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto;">
git clone git@github.com:zhangyurou/PI-t.git
cd PI-t</pre>

        <h2>Usage</h2>
        <p style="margin-bottom: 20px;">Please follow these steps to use our code:</p>
        
        <ol style="text-align: left; line-height: 1.8;">
            <li><strong>Data Preprocessing</strong>
                <p>Convert 'zarr' format to 'txt' format using 'readzarr.py' for easier handling.</p>
            </li>

            <li><strong>Data Collection</strong>
                <p>Use the trajectory delta to control the robot in our pybullet environment to obtain joint information ('q, dotq, tau'). Navigate to the "direct_data" folder and run "visual-multi-time.py". Note: Modify your data in "visual-multi-time.py" and update the output filename in "block_pushing_multimodal.py".</p>
            </li>

            <li><strong>Data Cleaning</strong>
                <p>Remove columns 1, 7, and 13 (zero columns) using 'delete_column_txt.py'.</p>
            </li>

            <li><strong>Calculate Joint Acceleration</strong>
                <p>Use 'qdd.py' to calculate joint acceleration (dotdotq). This will result in an 18-column data file containing q (6 columns), dq (6 columns), and ddq (6 columns).</p>
            </li>

            <li><strong>Torque Calculation</strong>
                <p>Calculate joint torques using MATLAB. Navigate to the touque_matlab folder, use tau_calculate.m with your 18-column data, adjust the row number, and run to obtain 6 columns of torques.</p>
            </li>

            <li><strong>Data Concatenation</strong>
                <p>Use "combine_data.py" to merge joint information (18 columns) with trajectory data (2 columns), then combine with torque data (6 columns) to create a complete 26-column dataset.</p>
            </li>

            <li><strong>Data Normalization</strong>
                <p>Normalize the dataset using "normalization.py" for better training performance.</p>
            </li>

            <li><strong>Data Split</strong>
                <p>Use "train_test_data.py" to split the data into training (80%) and testing (20%) sets.</p>
            </li>

            <li><strong>Model Training</strong>
                <p>Train the "dynamic with trajectory" neural network using "LSTM_4_mask.py".</p>
            </li>

            <li><strong>Model Testing</strong>
                <p>Evaluate the model's performance using "LSTM_4_test_mask.py".</p>
            </li>
        </ol>
    </div>
</div>