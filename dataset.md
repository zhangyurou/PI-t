---
layout: default
title: Dataset
---

<div class="dataset-page">
    <div class="dataset-sidebar">
        <ul class="dataset-nav">
            <li><a href="#block-pushing" class="active">Block-Pushing</a></li>
            <li><a href="#pi-tg">PI-TG</a></li>
            <li><a href="#another-dataset">Another Dataset</a></li>
        </ul>
    </div>

    <div class="dataset-content">
        <div id="block-pushing">
            <h1>Block-Pushing 数据集</h1>

            <h2>数据集下载</h2>
            <p>您可以通过以下链接下载我们的数据集：</p>
            <a href="https://github.com/zhangyurou/PI-t/blob/main/block_pushing.zip">下载数据集</a>

            <h2>数据集效果</h2>
            <p>以下是数据集的仿真环境：</p>
            <img src="{{ '/assets/images/block-pushing.png' | relative_url }}" alt="PI-TG Image" width="100">
            
            <h2>数据集统计</h2>
            <ul>
                <li>总样本数：15,000</li>
                <li>训练集：12,000</li>
                <li>测试集：3,000</li>
                <li>类别数：12</li>
            </ul>
        </div>

        <div id="pi-tg" style="display: none;">
            <h1>PI-TG 数据集</h1>
            <h2>数据集下载</h2>
            <p>您可以通过以下链接下载我们的数据集：</p>
            <a href="https://github.com/NVlabs/mimicgen#downloading-and-using-datasets">下载数据集</a>
            
            <h2>数据集效果</h2>
            <p>以下是数据集的仿真环境：</p>
            <img src="{{ '/assets/images/pi-tg.png' | relative_url }}" alt="PI-TG Image" width="100">
            
            <h2>数据集统计</h2>
            <ul>
                <li>总样本数：15,000</li>
                <li>训练集：12,000</li>
                <li>测试集：3,000</li>
                <li>类别数：12</li>
            </ul>
        </div>

        <div id="another-dataset" style="display: none;">
            <h1>Another Dataset</h1>
            <h2>数据集下载</h2>
            <p>您可以通过以下链接下载我们的数据集：</p>
            <a href="https://example.com/dataset.zip">下载数据集</a>
            
            <h2>数据集效果</h2>
            <p>以下是数据集的仿真环境：</p>
            <img src="{{ '/assets/images/another-dataset.png' | relative_url }}" alt="Another Dataset Image" width="100">
            
            <h2>数据集统计</h2>
            <ul>
                <li>总样本数：8,000</li>
                <li>训练集：6,400</li>
                <li>测试集：1,600</li>
                <li>类别数：8</li>
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


