---
layout: default
title: Dataset
---

<div class="dataset-page" style="display: flex; width: 100%;">
    <div class="dataset-sidebar" style="width: 200px; padding: 20px; margin-right: 20px;">
        <ul class="dataset-nav" style="list-style: none; padding: 0; text-align: left;">
            <li style="margin-bottom: 10px;"><a href="#block-pushing" class="active" style="text-decoration: none; color: #333;">Block-Pushing</a></li>
            <li style="margin-bottom: 10px;"><a href="#push-t" style="text-decoration: none; color: #333;">Push-T</a></li>
            <li style="margin-bottom: 10px;"><a href="#mimicgen" style="text-decoration: none; color: #333;">MimicGen</a></li>
        </ul>
    </div>

    <div class="dataset-content" style="flex: 1; max-width: 1000px; padding: 20px;">
        <div id="block-pushing">
            <h1 style="text-align: left;">Block-Pushing Dataset</h1>

            <h2 style="text-align: left;">Download Dataset</h2>
            <p style="text-align: left;">You can download our dataset through the following link:</p>
            <p style="text-align: left;">https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip</p>
            <p style="text-align: left;"><a href="https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip">download</a></p>

            <h2 style="text-align: left;">The effection of the dataset</h2>
            <p style="text-align: left;">Simulation Environment:</p>
            <div style="text-align: center;">
                <img src="{{ '/assets/images/blockpushing.png' | relative_url }}" alt="Block-Pushing Image" width="300">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <video width="500" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/blockpushing.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>

            <h2>Dataset Statistics</h2>
            <ul>
                <li>Total samples: 114,962</li>
                <li>Action space: 2D (x, y coordinates of the robot arm end-effector)</li>
                <li>Observation space: 16D (translation and rotation of two blocks, translation and rotation of two target positions, translation of the robot arm end-effector)</li>
            </ul>
        </div>


        <div id="push-t" style="display: none;">
            <h1>Push-T Dataset</h1>

            <h2>Download Dataset</h2>
            <p>You can download our dataset through the following link:</p>
            <p>https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip</p>
            <a href="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip">download</a>

            <h2>The effection of the dataset</h2>
            <p>Simulation Environment:</p>
            <div style="text-align: center;">
                <img src="{{ '/assets/images/pusht.png' | relative_url }}" alt="Block-Pushing Image" width="300">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <video width="300" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/pusht.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>

            <h2>Dataset Statistics</h2>
            <ul>
                <li>Total samples: 114,962</li>
                <li>Action space: 2D (x, y coordinates of the robot arm end-effector)</li>
                <li>Observation space: 16D (translation and rotation of two blocks, translation and rotation of two target positions, translation of the robot arm end-effector)</li>
            </ul>
        </div>

        <div id="mimicgen" style="display: none;">
            <h1>MimicGen Dataset</h1>

            <h2>Download Dataset</h2>
            <p>You can download our dataset through the following link:</p>
            <p>https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip</p>
            <a href="https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip">download</a>

            <h2>The effection of the dataset</h2>
            <p>Simulation Environment:</p>
            <div style="text-align: center; display: flex; justify-content: space-between;">
                <img src="{{ '/assets/images/mimicgen1.png' | relative_url }}" alt="Block-Pushing Image" width="200">
                <img src="{{ '/assets/images/mimicgen2.png' | relative_url }}" alt="Block-Pushing Image" width="200">
                <img src="{{ '/assets/images/mimicgen3.png' | relative_url }}" alt="Block-Pushing Image" width="200">
                <img src="{{ '/assets/images/mimicgen4.png' | relative_url }}" alt="Block-Pushing Image" width="200">
            </div>
            <div style="text-align: center; margin-top: 20px; display: flex; justify-content: space-between;">
                <video width="200" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/mimicgen1.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <video width="200" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/mimicgen2.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <video width="200" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/mimicgen3.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <video width="200" controls autoplay muted loop>
                    <source src="{{ '/assets/videos/mimicgen4.mp4' | relative_url }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>


            <h2>Dataset Statistics</h2>
            <ul>
                <li>Total samples: 114,962</li>
                <li>Action space: 2D (x, y coordinates of the robot arm end-effector)</li>
                <li>Observation space: 16D (translation and rotation of two blocks, translation and rotation of two target positions, translation of the robot arm end-effector)</li>
            </ul>
        </div>
    </div>
</div>

<script>
document.querySelectorAll('.dataset-nav a').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        // 隐藏所有内容
        document.querySelectorAll('.dataset-content > div').forEach(div => {
            div.style.display = 'none';
        });
        // 显示选中的内容
        document.querySelector(this.getAttribute('href')).style.display = 'block';
        // 更新active状态
        document.querySelectorAll('.dataset-nav a').forEach(a => {
            a.classList.remove('active');
        });
        this.classList.add('active');
    });
});
</script>

<footer style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f5f5f5;">
    <p>© Copyright Beijing Key Laboratory of Light Industrial Robotics and Safety Verification, College of Information Engineering, Capital Normal University 2025</p>
</footer>


