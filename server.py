# coding=utf-8
import os
import datetime
import tensorflow as tf
import numpy as np
from bilstm_model import BiLSTM
from bilstm_model import BiLSTMConfig
from data import preprocess

import asyncio
import websockets
import base64
from urllib import parse
import json
import codecs

print(">>>>>>>> loading trained model...")

model_list = [
    { 'name': 'CNN1', 'path': '' },
    { 'name': 'CNN2', 'path': '' },
    { 'name': 'CNN3', 'path': '' },
    { 'name': 'RNN1', 'path': '' },
    { 'name': 'RNN2', 'path': '' },
    { 'name': 'BiLSTM1', 'path': '' },
    { 'name': 'BiLSTM2', 'path': '' },
]

"""
读取模型，预测商品标题
:param titles: 列表，商品标题的字符串
:return: results
"""
# Test procedure
# ======================================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) # 不使用with，保持sess避免重复加载
# TODO: 读取不同模型，修改此处参数
# 要读取的模型路径
checkpoint_dir = os.path.abspath("checkpoints\\bilstm") # os.path.abspath("checkpoints\\textrnn")
# 模型的文件名放在这，不含后缀
checkpoint_file = os.path.join(checkpoint_dir, "WORD-NON-STATIC-76580") # "WORD-NON-STATIC-30001"
# 这要加.meta后缀
saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'WORD-NON-STATIC-76580.meta'))
saver.restore(sess, checkpoint_file)
graph = tf.get_default_graph()

# 注意：测试时，rnn_model.py中的Config参数要和读取的模型参数一致
config = BiLSTMConfig()
cnn = BiLSTM(config)
# 读取测试集及词汇表数据
dataset, next_element = cnn.prepare_test_data()

# 从图中读取变量
input_x = graph.get_operation_by_name("input_x").outputs[0]
dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
prediction = graph.get_operation_by_name("output/prediction").outputs[0]
training = graph.get_operation_by_name("training").outputs[0]

# 读取分类标签
label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))

def predict_step(batch_x):
    feed_dict = {
        input_x: batch_x,
        dropout_keep_prob: 1.0,
        training: False
    }
    pre = sess.run(prediction, feed_dict)
    return pre

def customize_predict(titles):
    # 自定义批量查询
    # ================================================================================
    batch_x = []
    if cnn.train_mode == 'CHAR-RANDOM' or cnn.train_mode == 'WORD-NON-STATIC':
        # 1.id
        for title in titles:
            batch_x.append(preprocess.to_id(title, cnn.vocab, cnn.train_mode))
    batch_x = np.stack(batch_x)
    pre = predict_step(batch_x)
    results = [label[x] for x in pre]
    final_results = list(zip(titles, results))
    return final_results
    # ====================================================================================

def predict_test_set():
    # 给测试集打标签
    # ====================================================================================
    # 跳过测试集的标题
    sess.run(next_element)
    i = 0
    t1 = datetime.datetime.now()
    while True:
        try:
            titles = sess.run(next_element)
            batch_x = cnn.convert_test_input(titles)
            predict_step(batch_x)
            i += 1
        except tf.errors.OutOfRangeError:
            break

    t2 = datetime.datetime.now()
    dt = (t2-t1).min

    print('查询总耗时: %fmin' % dt)
    print('平均每条耗时: %fmin' % (dt/i))
    # 450w条数据约15分钟
    # ==================================================================

def decodeRequest(data):
    print(f'> {data}')
    return json.loads(data)

def encodeResults(results):
    obj = {
        'cmd': 'result',
        'results': []
    }
    for r in results:
        # filter some fields, add time delta
        it = {
            'id': r['id'],
            'type': r['type'],
            'status': r['status'],
            'result': r['result'],
            'time': ''
        }
        # 可选参数：start，用于显示时间
        if 'start' in r:
            sec = (datetime.datetime.now() - r['start']).total_seconds()
            it['time'] = '%.3f 秒' % sec
            if 'size' in r and sec > 1: # 避免除零错误
                it['time'] += '\n%d 条/秒' % int(r['size'] / sec)
        obj['results'].append(it)
    return json.dumps(obj)

# 向客户端发回数据
async def send_back(ws, data):
    await ws.send(data)
    print(f"< {data}")

async def handle_text(websocket, item):
    text = item['title']
    print(f"> text:{text}")

    item['start'] = datetime.datetime.now()
    result_of_text = customize_predict([text])[0][1]
    item['status'] = 'success'
    item['result'] = result_of_text
    await send_back(websocket, encodeResults([item]))

async def handle_file(websocket, item):
    # 按行分割
    items_in_file = item['file_content'].split('\n')
    src_summary = '\n'.join(items_in_file[0:min(10, len(items_in_file))]) + '\n...'
    print(f"> file:{src_summary}")
    item_size = len(items_in_file)

    item['size'] = 0 # 文件要记录行数，用于计算速度
    item['result'] = f'共{item_size}行数据待分类'
    item['start'] = datetime.datetime.now() # 记录开始时间，用于计算使用时间
    await send_back(websocket, encodeResults([item]))

    # 每100行输入模型预测
    batch_size = 100
    batch_step = item_size // batch_size + (item_size % batch_size != 0)
    # 输出文件路径
    # results_path = 'outputs\\%s.results.tsv' % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    results_path = item['title'].split('.')
    results_path.insert(-1, 'results')
    results_path = os.path.join('outputs', '.'.join(results_path))
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    with codecs.open(results_path, 'w', encoding='gbk', errors='ignore') as resfile:
        for s in range(batch_step):
            start, end = s*batch_size, (s+1)*batch_size
            if s == batch_step - 1:
                end = item_size
            # 预测
            results_of_step = customize_predict(items_in_file[start:end])
            # 写入文件
            for j in range(len(results_of_step)):
                resfile.write(items_in_file[start+j])
                resfile.write('\t')
                resfile.write(results_of_step[j][1])
                resfile.write('\n')
            resfile.flush()
            # 更新进度
            progress = round(end / item_size * 100, 2)
            item['size'] += (end - start)
            item['status'] = 'pending'
            item['result'] = f'{end}/{item_size} {progress}%'
            await send_back(websocket, encodeResults([item]))
    # 分类完成
    item['status'] = 'success'
    item['result'] = f'{results_path}'
    await send_back(websocket, encodeResults([item]))

async def handle_item(websocket, item):
    if 'text' == item['type']:
        await handle_text(websocket, item)

    elif 'file' == item['type']:                
        # decode base 64 file content: -> base64 -> escape -> uriComponent
        try:
            src_data = parse.unquote(base64.b64decode(item['file_content']).decode('utf-8'))
            src_data = src_data.strip().replace('\r', '')
        except Exception as e:
            src_data = None
            print(e)
        if src_data:
            item['file_content'] = src_data
            await handle_file(websocket, item)
        else:
            item['status'] = 'failed'
            item['result'] = '解码出现异常'
            await send_back(websocket, encodeResults([item]))

    elif 'path' == item['type']:
        src_path = item['title']
        print(f"> path:{src_path}")
        if(os.path.exists(src_path)):
            src_data = None
            with codecs.open(src_path, 'r', encoding='gbk', errors='ignore') as src_file:
                src_data = src_file.read()
                src_data = src_data.strip().replace('\r', '')
            if src_data:
                item['file_content'] = src_data
                await handle_file(websocket, item)
            else:
                item['status'] = 'failed'
                item['result'] = f'文件打开失败{src_path}'
                await send_back(websocket, encodeResults([item]))
        else:
            item['status'] = 'failed'
            item['result'] = f'找不到文件：{src_path}'
            await send_back(websocket, encodeResults([item]))
    
async def demon(websocket, title):
    data = await websocket.recv()
    req = decodeRequest(data)


    # TODO: concurrency, cancle task
    # TODO: model select
    if 'query' == req['cmd']:
        # 更新为pending状态
        for item in req['items']:
            item['status'] = 'pending'
            item['result'] = '队列中...'
        res = encodeResults(req['items'])
        await send_back(websocket, res)

        for item in req['items']:
            await handle_item(websocket, item)

    elif 'clear' == req['cmd']:
        print(req)

    elif 'clearAll' == req['cmd']:
        print(req)

    elif 'modelList' == req['cmd']:
        await send_back(websocket, json.dumps({
            'cmd': 'modelList',
            'results': model_list
        }))
    
    elif 'ping' == req['cmd']:
        await send_back(websocket, json.dumps({ 'cmd': 'pong' }))


if __name__ == '__main__':
    print('>>>>>>>> server has listen on ws://localhost:8765')
    os.system("start demo.html")
    start_server = websockets.serve(demon, 'localhost', 8765)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()




