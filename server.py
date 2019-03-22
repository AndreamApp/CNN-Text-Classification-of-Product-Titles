# coding=utf-8
print(f'>>>>>>>> import libraries...')
from test import Predictor

import os
import asyncio
import websockets
import base64
from urllib import parse
import datetime
import json
import codecs

model = ['textcnn', 'textrnn', 'bilstm']
pred_mode = ['CHAR-RANDOM', 'WORD-NON-STATIC', 'MULTI']
model_dict = {
    'TextCNN-CHAR' : { 'model': model[0], 'pred_mode': pred_mode[0] },
    'TextCNN-WORD' : { 'model': model[0], 'pred_mode': pred_mode[1] },
    'TextCNN-MULTI' : { 'model': model[0], 'pred_mode': pred_mode[2] },
    'TextRNN-CHAR' : { 'model': model[1], 'pred_mode': pred_mode[0] },
    'TextRNN-WORD' : { 'model': model[1], 'pred_mode': pred_mode[1] },
    'BiLSTM-CHAR' : { 'model': model[2], 'pred_mode': pred_mode[0] },
    'BiLSTM-WORD' : { 'model': model[2], 'pred_mode': pred_mode[1] },
}

predictor = None

def predict(titles):
    return predictor.predict(titles)

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

async def query_text(item, send):
    text = item['title']
    print(f"> text:{text}")

    item['start'] = datetime.datetime.now()
    result_of_text = predict([text])[0][1]
    item['status'] = 'success'
    item['result'] = result_of_text
    return item

async def query_file(item, send):
    # 按行分割
    items_in_file = item['file_content'].split('\n')
    src_summary = '\n'.join(items_in_file[0:min(10, len(items_in_file))]) + '\n...'
    print(f"> file:{src_summary}")
    item_size = len(items_in_file)

    item['size'] = 0 # 文件要记录行数，用于计算速度
    item['result'] = f'共{item_size}行数据待分类'
    item['start'] = datetime.datetime.now() # 记录开始时间，用于计算使用时间
    await send([item])

    results_path = item['title'].split('.')
    results_path.insert(-1, 'results')
    results_path = os.path.join('outputs', '.'.join(results_path))
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # 每100行输入模型预测
    batch_size = 500
    batch_step = item_size // batch_size + (item_size % batch_size != 0)
    with codecs.open(results_path, 'w', encoding='gbk', errors='ignore') as resfile:
        # 输出文件路径
        # results_path = 'outputs\\%s.results.tsv' % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        for s in range(batch_step):
            start, end = s*batch_size, (s+1)*batch_size
            if s == batch_step - 1:
                end = item_size
            # 预测
            results_of_step = predict(items_in_file[start:end])
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
            await send([item])
    # 分类完成
    item['status'] = 'success'
    item['result'] = f'{results_path}'
    return item

async def query_item(item, send):
    if 'text' == item['type']:
        await query_text(item, send)

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
            await query_file(item, send)
        else:
            item['status'] = 'failed'
            item['result'] = '解码出现异常'

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
                await query_file(item, send)
            else:
                item['status'] = 'failed'
                item['result'] = f'文件打开失败{src_path}'
        else:
            item['status'] = 'failed'
            item['result'] = f'找不到文件：{src_path}'
    return item

async def query(items, send):
    # 更新为pending状态
    for item in items:
        item['status'] = 'pending'
        item['result'] = '队列中...'
    await send(items)

    results = []
    for item in items:
        res = await query_item(item, send)
        results.append(res)
    return results

async def clear(items, send):
    pass

async def clearAll(send):
    pass

async def serve(ws, title):
    data = await ws.recv()
    req = decodeRequest(data)

    res = None
    send = lambda items: send_back(ws, encodeResults(items))
    # TODO: concurrency, cancle task
    if 'query' == req['cmd']:
        if len(req['items']) > 0:
            model = req['model']
            predictor = predictor = model_dict[model]['predictor']
            res = await query(req['items'], send)

    elif 'clear' == req['cmd']:
        res = await clear(req['items'], send)

    elif 'clearAll' == req['cmd']:
        res = await clearAll(send)

    elif 'ping' == req['cmd']:
        await send_back(ws, json.dumps({ 'cmd': 'pong' }))
    
    if res:
        await send(res)

def listen(port):
    print(f'>>>>>>>> server has listen on ws://localhost:{port}')
    start_server = websockets.serve(serve, 'localhost', port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    for (name, params) in model_dict.items():
        pred = Predictor()
        pred.setModel(params['model'], params['pred_mode'])
        model_dict[name]['predictor'] = pred
        print(f'>>>>>>>> {name} loaded')
    
    predictor = model_dict['TextCNN-CHAR']['predictor']

    os.system("start demo.html")
    listen(8765)


