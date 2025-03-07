---
layout: default
title: Video
---

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 引入外部 CSS 文件 -->
    <link rel="stylesheet" href="{{ '/assets/css/styles.css' | relative_url }}">
</head>

<body>
    <div class="center-container">
      <div class="video-container">
        <h1>My Video</h1>
        <video width="1080" height="720" controls autoplay muted loop>
          <source src="{{ '/assets/videos/iros.mp4' | relative_url }}" type="video/mp4">
        </video>
      </div>
    </div>
</body>
<footer style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f5f5f5;">
    <p>© Copyright Beijing Key Laboratory of Light Industrial Robotics and Safety Verification, College of Information Engineering, Capital Normal University 2025</p>
</footer>
</html>


