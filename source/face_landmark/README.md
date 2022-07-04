## WFLW
- [https://wywu.github.io/projects/LAB/WFLW.html](https://wywu.github.io/projects/LAB/WFLW.html)   
- The format of txt ground truth list (7,500 for training and 2,500 for testing).
- coordinates of 98 landmarks (196)
- coordinates of upper left corner and lower right corner of detection rectangle (4)
- attributes annotations (6) + image name (1)
- attributes : pose, expression, illumination, make_up, occlusion, blur

## 300VW
- [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)  
- [300W-LP](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing&resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA)
- [300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)



## Record
- lr = 1e-3

1. custom scheduler  
    - val_loss : 1.56266

            def adjust_lr(epoch, lr):
                epoch+=1
                if epoch % 10 != 0:
                    return lr
                else:
                    return lr * 0.5

            optimizer = AngularGrad(method_angle="cos", learning_rate=lr)
            tf.keras.callbacks.LearningRateScheduler(adjust_lr),


2. CyclicalLearningRate(clr)
    - val_loss  : 2.33912(early stopped)

            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=0.000001,
                                                      maximal_learning_rate=0.01,
                                                      step_size=epochs / 2,
                                                      scale_fn=lambda x: 1.0,
                                                      scale_mode="cycle")

3. CosineDecayRestarts(cdr)
    - val_loss : 1.56002(early stopped)

            cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr,
                                                                    first_decay_steps=100,
                                                                    t_mul=2.0,
                                                                    m_mul=0.9,
                                                                    alpha=0.0001)

4. L2Loss
    - val_loss : 0.07569(early stopped)


            def L2Loss():
                def _L2Loss(y_true, y_pred):
                    landmarks, angle = y_pred[:, :136], y_pred[:, 136:]
                    landmark_gt, _, _ = tf.cast(y_true[:,:136], tf.float32),tf.cast(y_true[:,136:142], tf.float32),tf.cast(y_true[:,142:],tf.float32)
                    l2_distant = tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
                    
                    return tf.reduce_mean(l2_distant)

                return _L2Loss

            model.compile(loss={'train_out': L2Loss()}, optimizer=optimizer)