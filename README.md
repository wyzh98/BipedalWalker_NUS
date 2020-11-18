# BipedalWalker_NUS

It is a class project of #ME5406 Deep Learning for Robotics# @NUS

This project is built under [OpenAI gym](https://gym.openai.com/) environment, and Box2D physical engine, neural networks are deployed with Tensorflow 1.11.

>Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

By install the gym and its physical engine, you can try `pip install -e '.[box2d]' ` after cloning the latest gym environment.

Proximal  policy optimization (PPO) is implemented to solve the problem. Here are some instructions to run the files.

```
./events     # store the models .ckpt, tensorboard event files, and some gifs
models.py    # models of PPO, PPO with LSTM, A2C and PPO used for hardcore
                (most of them have the same structure, but copied for times)
PPO.py       # run this file to train PPO model
PPO_LSTM.py  # run this file to train PPO model with LSTM cell
A2C.py       # run this file to train A2C model (occasional bugs haven't fixed yet)
utils.py     # some utilities for statistics and discount
Validation_PPO.py           # run this file to validate restored PPO model
Validation_PPO_Hardcore.py  # run this file to validate another restored PPO
                                model under hardcore (not fully trained yet)
requirements.txt            # packages and their version of my Python environment, 
                            # some of them are not a necessity for this project.
video.mp4                   # video for demonstration
```

The robot learns to not tremble first, then try to crawl, and then learns to walk in a pretty weird way. After enough training, it learns to walk (even run).

![](https://github.com/wyzh98/BipedalWalker_NUS/blob/main/video.gif)





