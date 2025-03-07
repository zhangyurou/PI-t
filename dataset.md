---
layout: default
title: Dataset
---

<div class="dataset-page">
    <div class="dataset-sidebar">
        <ul class="dataset-nav">
            <li><a href="#block-pushing" class="active">Block-Pushing</a></li>
            <li><a href="#push-t">Push-T</a></li>
            <li><a href="#mimicgen">MimicGen</a></li>
        </ul>
    </div>

    <div class="dataset-content">
        <div id="block-pushing">
            <h1>Block-Pushing Dataset</h1>

            <h2>Download Dataset</h2>
            <p>You can download our dataset through the following link:</p>
            <p>https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip</p>
            <a href="https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip">downloading</a>

            <h2>The effection of the dataset</h2>
            <p>Simulation Environment:</p>
            <div style="text-align: center;">
                <img src="{{ '/assets/images/block-pushing.png' | relative_url }}" alt="Block-Pushing Image" width="300">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <video width="500" controls>
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
            <p>https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip</p>
            <a href="https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip">downloading</a>

            <h2>The effection of the dataset</h2>
            <p>Simulation Environment:</p>
            <div style="text-align: center;">
                <img src="{{ '/assets/images/block-pushing.png' | relative_url }}" alt="Block-Pushing Image" width="300">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <video width="500" controls>
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

        <div id="mimicgen" style="display: none;">
            <h1>Another Dataset</h1>

            <h2>Download Dataset</h2>
            <p>You can download our dataset through the following link:</p>
            <p>https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip</p>
            <a href="https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip">downloading</a>

            <h2>The effection of the dataset</h2>
            <p>Simulation Environment:</p>
            <div style="text-align: center;">
                <img src="{{ '/assets/images/block-pushing.png' | relative_url }}" alt="Block-Pushing Image" width="300">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <video width="500" controls>
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


