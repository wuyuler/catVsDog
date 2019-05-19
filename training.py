import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2
IMG_H = 208
IMG_W = 208
BATCH_SIZE = 16
CAPACITY = 128
MAX_STEP = 10000
learning_rate = 0.0001


def run_training():
    train_dir = "/home/yjt/Documents/CatVsDog/train/"
    logs_train_dir = "logs/"

    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)


    summary_op = tf.summary.merge_all()
    #移植模型需要的接口
    tf.get_variable_scope().reuse_variables()
    x = tf.placeholder(tf.float32, shape=[208, 208, 3],name='x')
    image = tf.reshape(x, [1, 208, 208, 3])
    logit = model.inference(image, 1, 2)
    logit = tf.nn.softmax(logit,name='logit')
    pre_num = tf.argmax(logit, 1, output_type='int32', name="output")

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])


            if step % 100 == 0:
                print(sess.run(train_logits))
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc))
                # summary_str = sess.run(summary_op)
                # train_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                # checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                # saver.save(sess, checkpoint_path, global_step=step)
                #保存pb文件
                # 保存pb文件,用于android移植
                # 保存训练好的模型
                # 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                output_node_names=['output'])
                with tf.gfile.FastGFile('logs/catVsDog-'+str(step)+'.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
                    f.write(output_graph_def.SerializeToString())
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# 评估模型
from PIL import Image
import matplotlib.pyplot as plt


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image(train):
    train_dir = "/home/yjt/Documents/CatVsDog/train/"
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        logs_train_dir = "logs/"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" % global_step)
            else:
                print("No checkpoint file found")
            x=sess.graph.get_tensor_by_name('x:0')
            logit=sess.graph.get_tensor_by_name('logit:0')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction,1)
            if max_index == 0:
                print("This is a cat with possibility %.6f" % prediction[:, 0])
            else:
                print("This is a dog with possibility %.6f" % prediction[:, 1])

def mytest():
    train_dir = "/home/yjt/Documents/CatVsDog/train/"
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver
    with tf.Session() as sess:
        sess.run(init)
        # 模型一
        saver = tf.train.import_meta_graph(os.getcwd() + '/logs/model.ckpt-199.meta')  # 载入模型结构
        saver.restore(sess, os.getcwd() + '/logs/model.ckpt-199')  # 载入模型参数
        graph = tf.get_default_graph()  # 计算图
        x = sess.graph.get_tensor_by_name('x:0')
        logit = sess.graph.get_tensor_by_name('logit:0')
        prediction = sess.run(logit, feed_dict={x: image_array})
        max_index = np.argmax(prediction, 1)
        if max_index == 0:
            print("This is a cat with possibility %.6f" % prediction[:, 0])
        else:
            print("This is a dog with possibility %.6f" % prediction[:, 1])
def test_pd():
    # train_dir = "/home/yjt/Documents/CatVsDog/test/"
    # train = input_data.getTestImage(train_dir)
    # image_array = get_one_image(train)
    image_dir="/home/yjt/Documents/CatVsDog/test/"
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open('logs/catVsDog.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # x_test = x_test.reshape(1, 28 * 28)
            input_x = sess.graph.get_tensor_by_name("x:0")
            output = sess.graph.get_tensor_by_name("output:0")
            # 对图片进行测试
            pre_num = sess.run(output, feed_dict={input_x: image_array})  # 利用训练好的模型预测结果
            print('模型预测结果为：', pre_num)
            if pre_num == 0:
                print("This is a cat " )
            else:
                print("This is a dog " )



#run_training()

#evaluate_one_image()
#mytest()
test_pd()
#run_training()

