# **BC-IFL: A Multi-modal Bias Correction Framework for Enhanced Image Forgery Localization**

![Framework](./Framework.png)

**Environment**

> ```
> conda create -n BC-IFL python=3.8
> conda activate BC-IFL
> cd /YOUR_PROJECT/path
> pip install -r requirements.txt
> ```

**Dataset**

- Prepare the txt file like the format as follow:
  > ```
  > /Image/img1.png,/Mask/mask1.png
  > /Image/img2.png,/Mask/mask2.png
  > ...
  > ```

- Prepare the prediction of IFL model and the text feature of TFE before training.
  > ```
  > python TFE.py
  > python text_feature_generator.py
  > ```

**Start training**
  > python train.py
  
**Evaluation**
  > python text.py

**Part of experiments results**

- You can download the prediction results [here](https://pan.baidu.com/s/1STbGYWpEJYk9D-mosiO4ZA?pwd=7jxx "提取码: 7jxx")
