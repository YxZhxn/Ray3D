## README

The `pose-detection` repo contains:
 - VideoPose
 - Poseformer

---

## How-to:

Here is a basic set of instructions on how to use this tool.

**1] docker image:**

https://code.aibee.cn/video_understanding/dockers/-/blob/master/Dockerfile.torch
 

**2] train and test:**

```$> python3 main.py --cfg cfg_videopose```

**3] test:**

```$> python3 main.py --cfg cfg_videopose --evaluate best_epoch.bin --timestamp xxxx```
